// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for binary operations:
//! AddLayer, SubLayer, MulBinaryLayer (McCormick and Middle relaxations).
//!
//! For binary ops z = f(x_a, x_b), CROWN backward returns (bounds_a, bounds_b).
//! Soundness property: for all (x_a, x_b) in input bounds,
//!   concrete_a.lower + concrete_b.lower <= f(x_a, x_b) <= concrete_a.upper + concrete_b.upper
//!
//! AddLayer and SubLayer are exact (linear ops) — CROWN-IBP equivalence holds.
//! MulBinaryLayer uses McCormick envelope or Middle relaxation — bounds are sound but not tight.
//!
//! Part of proof_coverage audit: binary ops had zero proptest soundness coverage.
//!
//! WALL-CLOCK POLICY FOR THIS FILE: every `#[ntest::timeout(..)]` below is a
//! HANG SENTINEL, not a performance assertion, and the walls are deliberately
//! far above these tests' isolated cost (well under a second each).
//!
//! They have to be. These tests participate in the `ny-test-utils` env lock --
//! either holding the shared half so a concurrent writer cannot leak
//! `NY_DENSE_BUDGET_MB` into them mid-run, or holding the exclusive half
//! themselves. Waiting on that lock is CORRECT behaviour, not a hang, and the
//! wait can be long: `margin_row`'s `root_build_bit_identical_across_conv_grain`
//! holds the exclusive half across a loop that runs over 60 seconds. A 10s wall
//! turns that legitimate wait into a spurious failure -- measured, 17 of 20
//! full-suite failures at --test-threads=8 were exactly this, with zero
//! remaining `crown=-inf` leaks.
//!
//! MEASURE BEFORE LOWERING THEM.

use crate::layers::binary_ops::{
    AddLayer, MaxBinaryLayer, MinBinaryLayer, MulBinaryLayer, SubLayer,
};
use crate::LinearBounds;
use crate::MulBinaryRelaxationMode;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// Tolerance for affine binary CROWN soundness (Add, Sub are exact linear ops).
const AFFINE_BINARY_TOLERANCE: f32 = 1e-4;

/// Tolerance for McCormick/Middle relaxation soundness.
/// Looser than affine because McCormick involves products of bounds and
/// directed rounding on f64→f32 downcast accumulates more error.
const MCCORMICK_TOLERANCE: f32 = 1e-3;

fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

// =============================================================================
// ADD BINARY CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// AddLayer CROWN backward soundness with identity incoming bounds.
    ///
    /// For C = A + B (linear), CROWN backward with identity incoming should
    /// produce bounds that match IBP exactly. Verifies both CROWN-IBP equivalence
    /// and sampling soundness.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_add_binary_crown_identity(
        la0 in -5.0f32..5.0, da0 in 0.01f32..3.0,
        la1 in -5.0f32..5.0, da1 in 0.01f32..3.0,
        lb0 in -5.0f32..5.0, db0 in 0.01f32..3.0,
        lb1 in -5.0f32..5.0, db1 in 0.01f32..3.0,
    ) {
        // Excluded from overlapping an env WRITER. The leak is specific and
        // known: `NY_DENSE_BUDGET_MB`, read process-globally by
        // `crown_memory::explicit_cpu_crown_dense_budget_bytes`. A concurrent
        // test setting it to 0 starves this one's CROWN into an IBP fallback,
        // which surfaces here as `crown=-inf` -- an enclosure violation that
        // is really a race. Observed failing at --test-threads=4 and =8.
        let _env = crate::tests::lock_env_shared();
        let ua0 = (la0 + da0).min(5.0);
        let ua1 = (la1 + da1).min(5.0);
        let ub0 = (lb0 + db0).min(5.0);
        let ub1 = (lb1 + db1).min(5.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = AddLayer;

        // IBP reference
        let ibp_output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(2);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // Combined CROWN bounds
        for i in 0..2 {
            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];

            // CROWN-IBP equivalence (Add is linear)
            prop_assert!(
                (crown_lower - ibp_output.lower()[[i]]).abs() <= AFFINE_BINARY_TOLERANCE,
                "Add CROWN-IBP lower mismatch at {i}: crown={crown_lower}, ibp={}",
                ibp_output.lower()[[i]]
            );
            prop_assert!(
                (crown_upper - ibp_output.upper()[[i]]).abs() <= AFFINE_BINARY_TOLERANCE,
                "Add CROWN-IBP upper mismatch at {i}: crown={crown_upper}, ibp={}",
                ibp_output.upper()[[i]]
            );
        }

        // Sampling soundness
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0 + xb0, xa1 + xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
                            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
                            prop_assert!(
                                yi >= crown_lower - AFFINE_BINARY_TOLERANCE,
                                "Add lower violation at {i}: y={yi} < lb={crown_lower}",
                            );
                            prop_assert!(
                                yi <= crown_upper + AFFINE_BINARY_TOLERANCE,
                                "Add upper violation at {i}: y={yi} > ub={crown_upper}",
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// SUB BINARY CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// SubLayer CROWN backward soundness with identity incoming bounds.
    ///
    /// For C = A - B (linear), CROWN backward negates B-branch coefficients
    /// and swaps lower/upper. Verifies CROWN-IBP equivalence and sampling soundness.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_sub_binary_crown_identity(
        la0 in -5.0f32..5.0, da0 in 0.01f32..3.0,
        la1 in -5.0f32..5.0, da1 in 0.01f32..3.0,
        lb0 in -5.0f32..5.0, db0 in 0.01f32..3.0,
        lb1 in -5.0f32..5.0, db1 in 0.01f32..3.0,
    ) {
        // Excluded from overlapping an env WRITER. The leak is specific and
        // known: `NY_DENSE_BUDGET_MB`, read process-globally by
        // `crown_memory::explicit_cpu_crown_dense_budget_bytes`. A concurrent
        // test setting it to 0 starves this one's CROWN into an IBP fallback,
        // which surfaces here as `crown=-inf` -- an enclosure violation that
        // is really a race. Observed failing at --test-threads=4 and =8.
        let _env = crate::tests::lock_env_shared();
        let ua0 = (la0 + da0).min(5.0);
        let ua1 = (la1 + da1).min(5.0);
        let ub0 = (lb0 + db0).min(5.0);
        let ub1 = (lb1 + db1).min(5.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = SubLayer;

        // IBP reference
        let ibp_output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(2);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // Combined CROWN bounds
        for i in 0..2 {
            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];

            // CROWN-IBP equivalence (Sub is linear)
            prop_assert!(
                (crown_lower - ibp_output.lower()[[i]]).abs() <= AFFINE_BINARY_TOLERANCE,
                "Sub CROWN-IBP lower mismatch at {i}: crown={crown_lower}, ibp={}",
                ibp_output.lower()[[i]]
            );
            prop_assert!(
                (crown_upper - ibp_output.upper()[[i]]).abs() <= AFFINE_BINARY_TOLERANCE,
                "Sub CROWN-IBP upper mismatch at {i}: crown={crown_upper}, ibp={}",
                ibp_output.upper()[[i]]
            );
        }

        // Sampling soundness
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0 - xb0, xa1 - xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
                            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
                            prop_assert!(
                                yi >= crown_lower - AFFINE_BINARY_TOLERANCE,
                                "Sub lower violation at {i}: y={yi} < lb={crown_lower}",
                            );
                            prop_assert!(
                                yi <= crown_upper + AFFINE_BINARY_TOLERANCE,
                                "Sub upper violation at {i}: y={yi} > ub={crown_upper}",
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// MUL BINARY MCCORMICK CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// MulBinaryLayer McCormick CROWN backward soundness with identity incoming.
    ///
    /// For z = x * y, McCormick envelope provides sound but not tight linear bounds.
    /// Verifies that concretized CROWN bounds contain the true product for all
    /// sampled (x, y) in the input box. Uses 2-element tensors to keep sampling
    /// tractable.
    ///
    /// Reference: McCormick (1976), "Computability of global solutions to factorable
    /// nonconvex programs". Implementation in mul/mod.rs:select_mccormick_plane.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_mccormick_crown_identity(
        la0 in -3.0f32..3.0, da0 in 0.01f32..2.0,
        la1 in -3.0f32..3.0, da1 in 0.01f32..2.0,
        lb0 in -3.0f32..3.0, db0 in 0.01f32..2.0,
        lb1 in -3.0f32..3.0, db1 in 0.01f32..2.0,
    ) {
        let ua0 = (la0 + da0).min(3.0);
        let ua1 = (la1 + da1).min(3.0);
        let ub0 = (lb0 + db0).min(3.0);
        let ub1 = (lb1 + db1).min(3.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MulBinaryLayer;

        // CROWN backward with McCormick relaxation
        let identity = LinearBounds::identity(2);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::McCormick,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (McCormick) failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // Combined CROWN bounds
        let crown_lower = [
            concrete_a.lower()[[0]] + concrete_b.lower()[[0]],
            concrete_a.lower()[[1]] + concrete_b.lower()[[1]],
        ];
        let crown_upper = [
            concrete_a.upper()[[0]] + concrete_b.upper()[[0]],
            concrete_a.upper()[[1]] + concrete_b.upper()[[1]],
        ];

        // Note: unlike linear ops, McCormick CROWN concretized per-branch can be
        // LOOSER than IBP for element-wise multiplication. IBP computes exact interval
        // products, while McCormick linearizes z = x*y and concretizes each branch
        // independently (sum of independent minimizations >= direct minimum of product).
        // So we only check soundness (true output in bounds), not tightness vs IBP.

        // Sampling soundness: true product must lie within CROWN bounds
        let spts = sample_points(0.0, 1.0, 7);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0 * xb0, xa1 * xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            prop_assert!(
                                yi >= crown_lower[i] - MCCORMICK_TOLERANCE,
                                "McCormick lower violation at {i}: y={yi} < lb={}, \
                                 xa=[{xa0},{xa1}], xb=[{xb0},{xb1}], \
                                 bounds_a=[{la0},{ua0}]x[{la1},{ua1}], \
                                 bounds_b=[{lb0},{ub0}]x[{lb1},{ub1}]",
                                crown_lower[i]
                            );
                            prop_assert!(
                                yi <= crown_upper[i] + MCCORMICK_TOLERANCE,
                                "McCormick upper violation at {i}: y={yi} > ub={}, \
                                 xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]",
                                crown_upper[i]
                            );
                        }
                    }
                }
            }
        }
    }

    /// MulBinaryLayer Middle relaxation CROWN backward soundness with identity incoming.
    ///
    /// Middle relaxation uses fixed r=0.5 interpolation coefficients matching
    /// auto_LiRPA's `mul.middle`. Must also be sound (bounds contain true product).
    ///
    /// Reference: auto_LiRPA/operators/bivariate.py:MulHelper.interpolated_relaxation
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_middle_crown_identity(
        la0 in -3.0f32..3.0, da0 in 0.01f32..2.0,
        la1 in -3.0f32..3.0, da1 in 0.01f32..2.0,
        lb0 in -3.0f32..3.0, db0 in 0.01f32..2.0,
        lb1 in -3.0f32..3.0, db1 in 0.01f32..2.0,
    ) {
        let ua0 = (la0 + da0).min(3.0);
        let ua1 = (la1 + da1).min(3.0);
        let ub0 = (lb0 + db0).min(3.0);
        let ub1 = (lb1 + db1).min(3.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MulBinaryLayer;

        // CROWN backward with Middle relaxation
        let identity = LinearBounds::identity(2);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::Middle,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (Middle) failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // Combined CROWN bounds
        let crown_lower = [
            concrete_a.lower()[[0]] + concrete_b.lower()[[0]],
            concrete_a.lower()[[1]] + concrete_b.lower()[[1]],
        ];
        let crown_upper = [
            concrete_a.upper()[[0]] + concrete_b.upper()[[0]],
            concrete_a.upper()[[1]] + concrete_b.upper()[[1]],
        ];

        // Sampling soundness: true product must lie within CROWN bounds
        let spts = sample_points(0.0, 1.0, 7);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0 * xb0, xa1 * xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            prop_assert!(
                                yi >= crown_lower[i] - MCCORMICK_TOLERANCE,
                                "Middle lower violation at {i}: y={yi} < lb={}, \
                                 xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]",
                                crown_lower[i]
                            );
                            prop_assert!(
                                yi <= crown_upper[i] + MCCORMICK_TOLERANCE,
                                "Middle upper violation at {i}: y={yi} > ub={}, \
                                 xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]",
                                crown_upper[i]
                            );
                        }
                    }
                }
            }
        }
    }

    /// MulBinaryLayer McCormick CROWN with non-identity incoming coefficients.
    ///
    /// Tests composition: k . (x * y) should be within CROWN bounds for all
    /// (x, y) in the input box. This exercises the weight-sign-dependent
    /// McCormick plane selection (positive k prefers tight lower planes,
    /// negative k prefers tight upper planes).
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_mccormick_crown_nonidentity(
        la0 in -2.0f32..2.0, da0 in 0.01f32..1.5,
        la1 in -2.0f32..2.0, da1 in 0.01f32..1.5,
        lb0 in -2.0f32..2.0, db0 in 0.01f32..1.5,
        lb1 in -2.0f32..2.0, db1 in 0.01f32..1.5,
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
    ) {
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01);

        let ua0 = (la0 + da0).min(2.0);
        let ua1 = (la1 + da1).min(2.0);
        let ub0 = (lb0 + db0).min(2.0);
        let ub1 = (lb1 + db1).min(2.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MulBinaryLayer;

        // Non-identity incoming: 1 output combining 2 inputs with coefficients k
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![k0, k1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![k0, k1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &incoming,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::McCormick,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (McCormick, non-identity) failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
        let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

        // Sampling soundness: k . (x * y)
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let combined = k0 * (xa0 * xb0) + k1 * (xa1 * xb1);

                        prop_assert!(
                            combined >= crown_lower - MCCORMICK_TOLERANCE,
                            "McCormick non-identity lower violation: k.(x*y)={combined} < lb={crown_lower}, \
                             k=[{k0},{k1}], xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]"
                        );
                        prop_assert!(
                            combined <= crown_upper + MCCORMICK_TOLERANCE,
                            "McCormick non-identity upper violation: k.(x*y)={combined} > ub={crown_upper}, \
                             k=[{k0},{k1}], xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]"
                        );
                    }
                }
            }
        }
    }

    /// MulBinaryLayer McCormick CROWN soundness for zero-crossing intervals.
    ///
    /// When input intervals cross zero (lower < 0, upper > 0), the McCormick
    /// envelope selection becomes more complex because different facets dominate
    /// in different quadrants. This test specifically targets zero-crossing to
    /// catch plane selection bugs.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_mccormick_zero_crossing(
        la0 in -3.0f32..-0.01, ra0 in 0.01f32..3.0,
        la1 in -3.0f32..-0.01, ra1 in 0.01f32..3.0,
        lb0 in -3.0f32..-0.01, rb0 in 0.01f32..3.0,
        lb1 in -3.0f32..-0.01, rb1 in 0.01f32..3.0,
    ) {
        // Force zero-crossing: lower < 0 < upper for both inputs
        let ua0 = ra0;
        let ua1 = ra1;
        let ub0 = rb0;
        let ub1 = rb1;

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MulBinaryLayer;

        let identity = LinearBounds::identity(2);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::McCormick,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (McCormick, zero-crossing) failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        let crown_lower = [
            concrete_a.lower()[[0]] + concrete_b.lower()[[0]],
            concrete_a.lower()[[1]] + concrete_b.lower()[[1]],
        ];
        let crown_upper = [
            concrete_a.upper()[[0]] + concrete_b.upper()[[0]],
            concrete_a.upper()[[1]] + concrete_b.upper()[[1]],
        ];

        // Test corners (extremes of the product) plus midpoints
        let spts = sample_points(0.0, 1.0, 5);
        for &ta in &spts {
            for &tb in &spts {
                let xa0 = la0 + ta * (ua0 - la0);
                let xa1 = la1 + ta * (ua1 - la1);
                let xb0 = lb0 + tb * (ub0 - lb0);
                let xb1 = lb1 + tb * (ub1 - lb1);
                let y = [xa0 * xb0, xa1 * xb1];

                for (i, &yi) in y.iter().enumerate() {
                    prop_assert!(
                        yi >= crown_lower[i] - MCCORMICK_TOLERANCE,
                        "Zero-crossing McCormick lower violation at {i}: y={yi} < lb={}, \
                         xa={xa0}, xb={xb0}",
                        crown_lower[i]
                    );
                    prop_assert!(
                        yi <= crown_upper[i] + MCCORMICK_TOLERANCE,
                        "Zero-crossing McCormick upper violation at {i}: y={yi} > ub={}, \
                         xa={xa0}, xb={xb0}",
                        crown_upper[i]
                    );
                }
            }
        }
    }
}

// =============================================================================
// MUL BINARY ALPHA-PARAMETERIZED CROWN BACKWARD SOUNDNESS (#3439 Phase 2)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Alpha at [0.5, 0.5] matches Middle mode (regression test).
    ///
    /// The interpolated McCormick with r_l=r_u=0.5 must produce identical
    /// coefficients to `compute_middle_coefficients`. Verifies that Phase 2
    /// correctly generalizes the existing Middle relaxation.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_alpha_half_matches_middle(
        la0 in -3.0f32..3.0, da0 in 0.01f32..2.0,
        la1 in -3.0f32..3.0, da1 in 0.01f32..2.0,
        lb0 in -3.0f32..3.0, db0 in 0.01f32..2.0,
        lb1 in -3.0f32..3.0, db1 in 0.01f32..2.0,
    ) {
        let ua0 = (la0 + da0).min(3.0);
        let ua1 = (la1 + da1).min(3.0);
        let ub0 = (lb0 + db0).min(3.0);
        let ub1 = (lb1 + db1).min(3.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MulBinaryLayer;
        let identity = LinearBounds::identity(2);

        // Middle mode (r=0.5 fixed)
        let (mid_a, mid_b) = layer.propagate_linear_binary(
            &identity, &input_a, &input_b,
            MulBinaryRelaxationMode::Middle,
        ).map_err(|e| TestCaseError::fail(
            format!("Middle failed: {e}")
        ))?;

        // Alpha mode with r_l=0.5, r_u=0.5
        let alphas = Array2::from_elem((2, 2), 0.5_f32);
        let (alpha_a, alpha_b) = layer.propagate_linear_binary_with_alpha(
            &identity, &input_a, &input_b,
            Some(&alphas),
        ).map_err(|e| TestCaseError::fail(
            format!("Alpha(0.5) failed: {e}")
        ))?;

        // Coefficients must match within directed rounding tolerance
        for i in 0..2 {
            for j in 0..2 {
                prop_assert!(
                    (mid_a.lower_a()[[i, j]] - alpha_a.lower_a()[[i, j]]).abs() < 1e-6,
                    "lower_a_a mismatch at [{i},{j}]: middle={}, alpha={}",
                    mid_a.lower_a()[[i, j]], alpha_a.lower_a()[[i, j]]
                );
                prop_assert!(
                    (mid_a.upper_a()[[i, j]] - alpha_a.upper_a()[[i, j]]).abs() < 1e-6,
                    "upper_a_a mismatch at [{i},{j}]: middle={}, alpha={}",
                    mid_a.upper_a()[[i, j]], alpha_a.upper_a()[[i, j]]
                );
                prop_assert!(
                    (mid_b.lower_a()[[i, j]] - alpha_b.lower_a()[[i, j]]).abs() < 1e-6,
                    "lower_a_b mismatch at [{i},{j}]: middle={}, alpha={}",
                    mid_b.lower_a()[[i, j]], alpha_b.lower_a()[[i, j]]
                );
                prop_assert!(
                    (mid_b.upper_a()[[i, j]] - alpha_b.upper_a()[[i, j]]).abs() < 1e-6,
                    "upper_a_b mismatch at [{i},{j}]: middle={}, alpha={}",
                    mid_b.upper_a()[[i, j]], alpha_b.upper_a()[[i, j]]
                );
            }
            prop_assert!(
                (mid_a.lower_b()[i] - alpha_a.lower_b()[i]).abs() < 1e-4,
                "lower_b mismatch at {i}: middle={}, alpha={}",
                mid_a.lower_b()[i], alpha_a.lower_b()[i]
            );
            prop_assert!(
                (mid_a.upper_b()[i] - alpha_a.upper_b()[i]).abs() < 1e-4,
                "upper_b mismatch at {i}: middle={}, alpha={}",
                mid_a.upper_b()[i], alpha_a.upper_b()[i]
            );
        }
    }

    /// Random alpha in [0, 1] always produces sound bounds (#3439 Phase 2).
    ///
    /// For any r_l, r_u in [0, 1], the interpolated McCormick envelope is a valid
    /// (sound) relaxation of z = x * y. Verifies by sampling (x, y) in the input
    /// box and checking that the true product lies within concretized bounds.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_alpha_random_sound(
        la0 in -3.0f32..3.0, da0 in 0.01f32..2.0,
        la1 in -3.0f32..3.0, da1 in 0.01f32..2.0,
        lb0 in -3.0f32..3.0, db0 in 0.01f32..2.0,
        lb1 in -3.0f32..3.0, db1 in 0.01f32..2.0,
        r_l0 in 0.0f32..1.0, r_l1 in 0.0f32..1.0,
        r_u0 in 0.0f32..1.0, r_u1 in 0.0f32..1.0,
    ) {
        let ua0 = (la0 + da0).min(3.0);
        let ua1 = (la1 + da1).min(3.0);
        let ub0 = (lb0 + db0).min(3.0);
        let ub1 = (lb1 + db1).min(3.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MulBinaryLayer;
        let identity = LinearBounds::identity(2);

        let alphas = Array2::from_shape_vec(
            (2, 2),
            vec![r_l0, r_l1, r_u0, r_u1],
        ).unwrap();

        let (bounds_a, bounds_b) = layer.propagate_linear_binary_with_alpha(
            &identity, &input_a, &input_b,
            Some(&alphas),
        ).map_err(|e| TestCaseError::fail(
            format!("Alpha random failed: {e}")
        ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        let crown_lower = [
            concrete_a.lower()[[0]] + concrete_b.lower()[[0]],
            concrete_a.lower()[[1]] + concrete_b.lower()[[1]],
        ];
        let crown_upper = [
            concrete_a.upper()[[0]] + concrete_b.upper()[[0]],
            concrete_a.upper()[[1]] + concrete_b.upper()[[1]],
        ];

        let spts = sample_points(0.0, 1.0, 7);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0 * xb0, xa1 * xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            prop_assert!(
                                yi >= crown_lower[i] - MCCORMICK_TOLERANCE,
                                "Alpha random lower violation at {i}: y={yi} < lb={}, \
                                 r_l=[{r_l0},{r_l1}], r_u=[{r_u0},{r_u1}]",
                                crown_lower[i]
                            );
                            prop_assert!(
                                yi <= crown_upper[i] + MCCORMICK_TOLERANCE,
                                "Alpha random upper violation at {i}: y={yi} > ub={}, \
                                 r_l=[{r_l0},{r_l1}], r_u=[{r_u0},{r_u1}]",
                                crown_upper[i]
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// MUL BINARY BROADCAST CROWN BACKWARD SOUNDNESS (#3499 proof coverage gap)
// =============================================================================

/// Helper to build broadcast BoundedTensors with shape [2, 3] (LHS) and [2, 1] (RHS).
/// This is the SE-block pattern where the scalar gate broadcasts across time dim.
fn make_broadcast_bt(
    la: &[f32; 6],
    ua: &[f32; 6],
    lb: &[f32; 2],
    ub: &[f32; 2],
) -> (BoundedTensor, BoundedTensor) {
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), la.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), ua.to_vec()).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), lb.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), ub.to_vec()).unwrap(),
    )
    .unwrap();
    (a, b)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// MulBinaryLayer McCormick CROWN broadcast soundness: [2,3] * [2,1].
    ///
    /// The non-alpha McCormick path uses the same `+=` accumulation for broadcast
    /// coefficient reduction as the alpha path, but had zero proptest coverage
    /// for broadcast shapes before this test. Verifies that the broadcast product
    /// z_j = a_j * b_{j//3} lies within the CROWN bounds for all sampled inputs.
    ///
    /// Gap identified by proof_coverage audit: P1 iteration 592.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_mccormick_broadcast_se_block(
        la0 in -2.0f32..2.0, da0 in 0.01f32..1.5,
        la1 in -2.0f32..2.0, da1 in 0.01f32..1.5,
        la2 in -2.0f32..2.0, da2 in 0.01f32..1.5,
        la3 in -2.0f32..2.0, da3 in 0.01f32..1.5,
        la4 in -2.0f32..2.0, da4 in 0.01f32..1.5,
        la5 in -2.0f32..2.0, da5 in 0.01f32..1.5,
        lb0 in -2.0f32..2.0, db0 in 0.01f32..1.5,
        lb1 in -2.0f32..2.0, db1 in 0.01f32..1.5,
    ) {
        let la = [la0, la1, la2, la3, la4, la5];
        let ua = [
            (la0 + da0).min(2.0), (la1 + da1).min(2.0), (la2 + da2).min(2.0),
            (la3 + da3).min(2.0), (la4 + da4).min(2.0), (la5 + da5).min(2.0),
        ];
        let lb = [lb0, lb1];
        let ub = [(lb0 + db0).min(2.0), (lb1 + db1).min(2.0)];

        let (input_a, input_b) = make_broadcast_bt(&la, &ua, &lb, &ub);
        let layer = MulBinaryLayer;

        // Non-identity spec: weighted combination of the 6 broadcast outputs.
        let spec = LinearBounds::new(
            Array2::from_shape_vec((1, 6), vec![1.0, -0.5, 0.25, -1.0, 0.75, 2.0]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 6), vec![1.0, -0.5, 0.25, -1.0, 0.75, 2.0]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &spec, &input_a, &input_b,
            MulBinaryRelaxationMode::McCormick,
        ).map_err(|e| TestCaseError::fail(
            format!("McCormick broadcast failed: {e}")
        ))?;

        // RHS coefficients must be reduced to [2] (the true input width of b)
        prop_assert_eq!(
            bounds_b.num_inputs(), 2,
            "RHS coefficients not reduced to broadcast source shape"
        );

        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);
        let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
        let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

        // Broadcast product: z_j = a_j * b_{j/3} for j in 0..6
        // Weighted sum: w . z
        let w = [1.0_f32, -0.5, 0.25, -1.0, 0.75, 2.0];
        let spts = sample_points(0.0, 1.0, 3);
        // 3^8 = 6561 evaluations
        for &ta0 in &spts { for &ta1 in &spts { for &ta2 in &spts {
            for &ta3 in &spts { for &ta4 in &spts { for &ta5 in &spts {
                for &tb0 in &spts { for &tb1 in &spts {
                    let xa = [
                        la[0] + ta0 * (ua[0] - la[0]),
                        la[1] + ta1 * (ua[1] - la[1]),
                        la[2] + ta2 * (ua[2] - la[2]),
                        la[3] + ta3 * (ua[3] - la[3]),
                        la[4] + ta4 * (ua[4] - la[4]),
                        la[5] + ta5 * (ua[5] - la[5]),
                    ];
                    let xb = [
                        lb[0] + tb0 * (ub[0] - lb[0]),
                        lb[1] + tb1 * (ub[1] - lb[1]),
                    ];
                    // Broadcast: positions 0,1,2 use xb[0]; positions 3,4,5 use xb[1]
                    let z: f32 = (0..6).map(|j| w[j] * xa[j] * xb[j / 3]).sum();

                    prop_assert!(
                        z >= crown_lower - MCCORMICK_TOLERANCE,
                        "Broadcast McCormick lower violation: z={z} < lb={crown_lower}, \
                         xa={xa:?}, xb={xb:?}"
                    );
                    prop_assert!(
                        z <= crown_upper + MCCORMICK_TOLERANCE,
                        "Broadcast McCormick upper violation: z={z} > ub={crown_upper}, \
                         xa={xa:?}, xb={xb:?}"
                    );
                }}
            }}
        }}}}
    }

    /// MulBinaryLayer alpha CROWN broadcast soundness: [2,3] * [2,1] with random alpha.
    ///
    /// Extends the alpha-parameterized McCormick proptest to the broadcast pattern.
    /// For any r_l, r_u in [0, 1], the interpolated McCormick with broadcast `+=`
    /// accumulation must produce sound bounds.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_mul_binary_alpha_broadcast_se_block(
        la0 in -2.0f32..2.0, da0 in 0.01f32..1.5,
        la1 in -2.0f32..2.0, da1 in 0.01f32..1.5,
        la2 in -2.0f32..2.0, da2 in 0.01f32..1.5,
        la3 in -2.0f32..2.0, da3 in 0.01f32..1.5,
        la4 in -2.0f32..2.0, da4 in 0.01f32..1.5,
        la5 in -2.0f32..2.0, da5 in 0.01f32..1.5,
        lb0 in -2.0f32..2.0, db0 in 0.01f32..1.5,
        lb1 in -2.0f32..2.0, db1 in 0.01f32..1.5,
        r_l0 in 0.0f32..1.0, r_l1 in 0.0f32..1.0,
        r_l2 in 0.0f32..1.0, r_l3 in 0.0f32..1.0,
        r_l4 in 0.0f32..1.0, r_l5 in 0.0f32..1.0,
    ) {
        let la = [la0, la1, la2, la3, la4, la5];
        let ua = [
            (la0 + da0).min(2.0), (la1 + da1).min(2.0), (la2 + da2).min(2.0),
            (la3 + da3).min(2.0), (la4 + da4).min(2.0), (la5 + da5).min(2.0),
        ];
        let lb = [lb0, lb1];
        let ub = [(lb0 + db0).min(2.0), (lb1 + db1).min(2.0)];

        let (input_a, input_b) = make_broadcast_bt(&la, &ua, &lb, &ub);
        let layer = MulBinaryLayer;

        let spec = LinearBounds::new(
            Array2::from_shape_vec((1, 6), vec![1.0, -0.5, 0.25, -1.0, 0.75, 2.0]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 6), vec![1.0, -0.5, 0.25, -1.0, 0.75, 2.0]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        // Alpha values: 2 rows (lower/upper) x 6 columns (broadcast output positions)
        // Use random r_l for lower row, 1-r_l for upper row (arbitrary choice)
        let alphas = Array2::from_shape_vec(
            (2, 6),
            vec![r_l0, r_l1, r_l2, r_l3, r_l4, r_l5,
                 1.0 - r_l0, 1.0 - r_l1, 1.0 - r_l2,
                 1.0 - r_l3, 1.0 - r_l4, 1.0 - r_l5],
        ).unwrap();

        let (bounds_a, bounds_b) = layer.propagate_linear_binary_with_alpha(
            &spec, &input_a, &input_b, Some(&alphas),
        ).map_err(|e| TestCaseError::fail(
            format!("Alpha broadcast failed: {e}")
        ))?;

        prop_assert_eq!(
            bounds_b.num_inputs(), 2,
            "Alpha broadcast RHS not reduced to [2]"
        );

        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);
        let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
        let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

        let w = [1.0_f32, -0.5, 0.25, -1.0, 0.75, 2.0];
        let spts = sample_points(0.0, 1.0, 3);
        for &ta0 in &spts { for &ta1 in &spts { for &ta2 in &spts {
            for &ta3 in &spts { for &ta4 in &spts { for &ta5 in &spts {
                for &tb0 in &spts { for &tb1 in &spts {
                    let xa = [
                        la[0] + ta0 * (ua[0] - la[0]),
                        la[1] + ta1 * (ua[1] - la[1]),
                        la[2] + ta2 * (ua[2] - la[2]),
                        la[3] + ta3 * (ua[3] - la[3]),
                        la[4] + ta4 * (ua[4] - la[4]),
                        la[5] + ta5 * (ua[5] - la[5]),
                    ];
                    let xb = [
                        lb[0] + tb0 * (ub[0] - lb[0]),
                        lb[1] + tb1 * (ub[1] - lb[1]),
                    ];
                    let z: f32 = (0..6).map(|j| w[j] * xa[j] * xb[j / 3]).sum();

                    prop_assert!(
                        z >= crown_lower - MCCORMICK_TOLERANCE,
                        "Alpha broadcast lower violation: z={z} < lb={crown_lower}"
                    );
                    prop_assert!(
                        z <= crown_upper + MCCORMICK_TOLERANCE,
                        "Alpha broadcast upper violation: z={z} > ub={crown_upper}"
                    );
                }}
            }}
        }}}}
    }
}

// =============================================================================
// MIN / MAX BINARY CROWN BACKWARD SOUNDNESS (exact convex-hull relaxation)
// =============================================================================

/// Tolerance for the piecewise-linear min/max convex-hull relaxation.
const MINMAX_TOLERANCE: f32 = 1e-3;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(400) })]

    /// MaxBinary CROWN backward soundness for identity and negated specs.
    ///
    /// The negated spec (-I) drives the w < 0 plane-selection branch, which is
    /// the soundness-critical path (uses the opposite envelope). For each output
    /// we verify the concretized affine bound encloses `sign * max(x, y)`.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_max_binary_crown(
        la0 in -3.0f32..3.0, da0 in 0.0f32..3.0,
        la1 in -3.0f32..3.0, da1 in 0.0f32..3.0,
        lb0 in -3.0f32..3.0, db0 in 0.0f32..3.0,
        lb1 in -3.0f32..3.0, db1 in 0.0f32..3.0,
    ) {
        let ua0 = la0 + da0;
        let ua1 = la1 + da1;
        let ub0 = lb0 + db0;
        let ub1 = lb1 + db1;
        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);
        let layer = MaxBinaryLayer;

        for &sign in &[1.0f32, -1.0f32] {
            let spec = LinearBounds::new(
                Array2::from_diag(&Array1::from_vec(vec![sign, sign])),
                Array1::zeros(2),
                Array2::from_diag(&Array1::from_vec(vec![sign, sign])),
                Array1::zeros(2),
            ).map_err(|e| TestCaseError::fail(format!("spec build failed: {e}")))?;
            let (bounds_a, bounds_b) = layer.propagate_linear_binary(&spec, &input_a, &input_b)
                .map_err(|e| TestCaseError::fail(format!("max propagate_linear_binary failed: {e}")))?;
            let concrete_a = bounds_a.concretize(&input_a);
            let concrete_b = bounds_b.concretize(&input_b);
            let crown_lower = [
                concrete_a.lower()[[0]] + concrete_b.lower()[[0]],
                concrete_a.lower()[[1]] + concrete_b.lower()[[1]],
            ];
            let crown_upper = [
                concrete_a.upper()[[0]] + concrete_b.upper()[[0]],
                concrete_a.upper()[[1]] + concrete_b.upper()[[1]],
            ];
            let spts = sample_points(0.0, 1.0, 7);
            for &ta0 in &spts { for &ta1 in &spts { for &tb0 in &spts { for &tb1 in &spts {
                let xa = [la0 + ta0 * (ua0 - la0), la1 + ta1 * (ua1 - la1)];
                let xb = [lb0 + tb0 * (ub0 - lb0), lb1 + tb1 * (ub1 - lb1)];
                let z = [sign * xa[0].max(xb[0]), sign * xa[1].max(xb[1])];
                for i in 0..2 {
                    prop_assert!(z[i] >= crown_lower[i] - MINMAX_TOLERANCE,
                        "Max lower violation i={i} sign={sign}: z={} < lb={}", z[i], crown_lower[i]);
                    prop_assert!(z[i] <= crown_upper[i] + MINMAX_TOLERANCE,
                        "Max upper violation i={i} sign={sign}: z={} > ub={}", z[i], crown_upper[i]);
                }
            }}}}
        }
    }

    /// MinBinary CROWN backward soundness for identity and negated specs.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_min_binary_crown(
        la0 in -3.0f32..3.0, da0 in 0.0f32..3.0,
        la1 in -3.0f32..3.0, da1 in 0.0f32..3.0,
        lb0 in -3.0f32..3.0, db0 in 0.0f32..3.0,
        lb1 in -3.0f32..3.0, db1 in 0.0f32..3.0,
    ) {
        let ua0 = la0 + da0;
        let ua1 = la1 + da1;
        let ub0 = lb0 + db0;
        let ub1 = lb1 + db1;
        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);
        let layer = MinBinaryLayer;

        for &sign in &[1.0f32, -1.0f32] {
            let spec = LinearBounds::new(
                Array2::from_diag(&Array1::from_vec(vec![sign, sign])),
                Array1::zeros(2),
                Array2::from_diag(&Array1::from_vec(vec![sign, sign])),
                Array1::zeros(2),
            ).map_err(|e| TestCaseError::fail(format!("spec build failed: {e}")))?;
            let (bounds_a, bounds_b) = layer.propagate_linear_binary(&spec, &input_a, &input_b)
                .map_err(|e| TestCaseError::fail(format!("min propagate_linear_binary failed: {e}")))?;
            let concrete_a = bounds_a.concretize(&input_a);
            let concrete_b = bounds_b.concretize(&input_b);
            let crown_lower = [
                concrete_a.lower()[[0]] + concrete_b.lower()[[0]],
                concrete_a.lower()[[1]] + concrete_b.lower()[[1]],
            ];
            let crown_upper = [
                concrete_a.upper()[[0]] + concrete_b.upper()[[0]],
                concrete_a.upper()[[1]] + concrete_b.upper()[[1]],
            ];
            let spts = sample_points(0.0, 1.0, 7);
            for &ta0 in &spts { for &ta1 in &spts { for &tb0 in &spts { for &tb1 in &spts {
                let xa = [la0 + ta0 * (ua0 - la0), la1 + ta1 * (ua1 - la1)];
                let xb = [lb0 + tb0 * (ub0 - lb0), lb1 + tb1 * (ub1 - lb1)];
                let z = [sign * xa[0].min(xb[0]), sign * xa[1].min(xb[1])];
                for i in 0..2 {
                    prop_assert!(z[i] >= crown_lower[i] - MINMAX_TOLERANCE,
                        "Min lower violation i={i} sign={sign}: z={} < lb={}", z[i], crown_lower[i]);
                    prop_assert!(z[i] <= crown_upper[i] + MINMAX_TOLERANCE,
                        "Min upper violation i={i} sign={sign}: z={} > ub={}", z[i], crown_upper[i]);
                }
            }}}}
        }
    }
}

// Div/Min/Max IBP soundness tests moved to ibp_binary_ops.rs (file size split).
