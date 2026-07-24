// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for ConcatLayer.
//!
//! Concat Y = concat(A, B) is a linear operation (element permutation/stacking).
//! CROWN backward splits the coefficient matrix at the boundary:
//!   coeffs[:, :size_a] → bounds_a, coeffs[:, size_a:] → bounds_b
//! with bias halved using directed rounding (#2173).
//!
//! Soundness property: for all (a, b) in input bounds,
//!   crown_lower[i] <= concat(a, b)[i] <= crown_upper[i]
//!
//! As a linear op, CROWN-IBP equivalence should hold (within FP tolerance).
//!
//! Part of #3104: last remaining binary ops CROWN proptest gap (Concat).

use crate::layers::binary_ops::ConcatLayer;
use crate::LinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// Tolerance for Concat CROWN soundness (linear op, same as Add/Sub).
const CONCAT_TOLERANCE: f32 = 1e-4;

fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

// =============================================================================
// CONCAT CROWN BACKWARD SOUNDNESS — SAME-SIZE INPUTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ConcatLayer CROWN backward soundness with identity incoming, same-size inputs.
    ///
    /// For Y = concat(A, B) with A: [2], B: [2], output: [4].
    /// Concat is linear so CROWN-IBP equivalence should hold.
    /// Verifies both CROWN-IBP equivalence and sampling soundness.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_concat_crown_identity_same_size(
        la0 in -5.0f32..5.0, da0 in 0.01f32..3.0,
        la1 in -5.0f32..5.0, da1 in 0.01f32..3.0,
        lb0 in -5.0f32..5.0, db0 in 0.01f32..3.0,
        lb1 in -5.0f32..5.0, db1 in 0.01f32..3.0,
    ) {
        let ua0 = (la0 + da0).min(5.0);
        let ua1 = (la1 + da1).min(5.0);
        let ub0 = (lb0 + db0).min(5.0);
        let ub1 = (lb1 + db1).min(5.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = ConcatLayer::new(0);

        // IBP reference: concat([la0,la1], [lb0,lb1]) → [la0,la1,lb0,lb1]
        let ibp_output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // CROWN backward: identity for 4-element output
        let identity = LinearBounds::identity(4);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity,
            &[2],  // input_a_shape
            &[2],  // input_b_shape
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // Combined CROWN bounds for each of the 4 output elements
        for i in 0..4 {
            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];

            // CROWN-IBP equivalence (Concat is linear)
            prop_assert!(
                (crown_lower - ibp_output.lower()[[i]]).abs() <= CONCAT_TOLERANCE,
                "Concat CROWN-IBP lower mismatch at {i}: crown={crown_lower}, ibp={}",
                ibp_output.lower()[[i]]
            );
            prop_assert!(
                (crown_upper - ibp_output.upper()[[i]]).abs() <= CONCAT_TOLERANCE,
                "Concat CROWN-IBP upper mismatch at {i}: crown={crown_upper}, ibp={}",
                ibp_output.upper()[[i]]
            );
        }

        // Sampling soundness: concat(a, b) must be within CROWN bounds
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        // True concat output: [xa0, xa1, xb0, xb1]
                        let y = [xa0, xa1, xb0, xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
                            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
                            prop_assert!(
                                yi >= crown_lower - CONCAT_TOLERANCE,
                                "Concat lower violation at {i}: y={yi} < lb={crown_lower}",
                            );
                            prop_assert!(
                                yi <= crown_upper + CONCAT_TOLERANCE,
                                "Concat upper violation at {i}: y={yi} > ub={crown_upper}",
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// CONCAT CROWN BACKWARD SOUNDNESS — DIFFERENT-SIZE INPUTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ConcatLayer CROWN backward soundness with different-size inputs.
    ///
    /// For Y = concat(A, B) with A: [3], B: [2], output: [5].
    /// Tests the asymmetric split path — coefficient matrix columns [0..3] → A,
    /// columns [3..5] → B.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_concat_crown_identity_different_sizes(
        la0 in -5.0f32..5.0, da0 in 0.01f32..3.0,
        la1 in -5.0f32..5.0, da1 in 0.01f32..3.0,
        la2 in -5.0f32..5.0, da2 in 0.01f32..3.0,
        lb0 in -5.0f32..5.0, db0 in 0.01f32..3.0,
        lb1 in -5.0f32..5.0, db1 in 0.01f32..3.0,
    ) {
        let ua0 = (la0 + da0).min(5.0);
        let ua1 = (la1 + da1).min(5.0);
        let ua2 = (la2 + da2).min(5.0);
        let ub0 = (lb0 + db0).min(5.0);
        let ub1 = (lb1 + db1).min(5.0);

        let input_a = make_bt(&[la0, la1, la2], &[ua0, ua1, ua2]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = ConcatLayer::new(0);

        // IBP reference
        let ibp_output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // CROWN backward: identity for 5-element output
        let identity = LinearBounds::identity(5);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity,
            &[3],  // input_a_shape
            &[2],  // input_b_shape
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // CROWN-IBP equivalence
        for i in 0..5 {
            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];

            prop_assert!(
                (crown_lower - ibp_output.lower()[[i]]).abs() <= CONCAT_TOLERANCE,
                "Concat asym CROWN-IBP lower mismatch at {i}: crown={crown_lower}, ibp={}",
                ibp_output.lower()[[i]]
            );
            prop_assert!(
                (crown_upper - ibp_output.upper()[[i]]).abs() <= CONCAT_TOLERANCE,
                "Concat asym CROWN-IBP upper mismatch at {i}: crown={crown_upper}, ibp={}",
                ibp_output.upper()[[i]]
            );
        }

        // Sampling soundness
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta2 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + 0.5 * (ua1 - la1); // fix middle for dim reduction
                        let xa2 = la2 + ta2 * (ua2 - la2);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        // True concat output: [xa0, xa1, xa2, xb0, xb1]
                        let y = [xa0, xa1, xa2, xb0, xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
                            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
                            prop_assert!(
                                yi >= crown_lower - CONCAT_TOLERANCE,
                                "Concat asym lower violation at {i}: y={yi} < lb={crown_lower}",
                            );
                            prop_assert!(
                                yi <= crown_upper + CONCAT_TOLERANCE,
                                "Concat asym upper violation at {i}: y={yi} > ub={crown_upper}",
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// CONCAT CROWN BACKWARD SOUNDNESS — NON-IDENTITY INCOMING
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ConcatLayer CROWN backward with non-identity incoming coefficients.
    ///
    /// Tests composition: k . concat(A, B) where k is a weight vector.
    /// This exercises the coefficient splitting when the incoming matrix
    /// is not identity — the split must correctly partition the weighted sum.
    ///
    /// With A: [2], B: [2], output: [4], we apply k: [1, 4] × [4] → [1]
    /// representing a weighted sum of the concat output.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_concat_crown_nonidentity(
        la0 in -3.0f32..3.0, da0 in 0.01f32..2.0,
        la1 in -3.0f32..3.0, da1 in 0.01f32..2.0,
        lb0 in -3.0f32..3.0, db0 in 0.01f32..2.0,
        lb1 in -3.0f32..3.0, db1 in 0.01f32..2.0,
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
        k2 in -2.0f32..2.0,
        k3 in -2.0f32..2.0,
    ) {
        // Ensure at least one non-trivial weight
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01 || k2.abs() > 0.01 || k3.abs() > 0.01);

        let ua0 = (la0 + da0).min(3.0);
        let ua1 = (la1 + da1).min(3.0);
        let ub0 = (lb0 + db0).min(3.0);
        let ub1 = (lb1 + db1).min(3.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = ConcatLayer::new(0);

        // Non-identity incoming: 1 output = weighted sum of 4 concat outputs
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), vec![k0, k1, k2, k3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![k0, k1, k2, k3]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &incoming,
            &[2],
            &[2],
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (non-identity) failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
        let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

        // Sampling soundness: k . concat(a, b)
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        // k . concat(a, b) = k0*xa0 + k1*xa1 + k2*xb0 + k3*xb1
                        let combined = k0 * xa0 + k1 * xa1 + k2 * xb0 + k3 * xb1;

                        prop_assert!(
                            combined >= crown_lower - CONCAT_TOLERANCE,
                            "Concat non-identity lower violation: k.y={combined} < lb={crown_lower}, \
                             k=[{k0},{k1},{k2},{k3}], xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]"
                        );
                        prop_assert!(
                            combined <= crown_upper + CONCAT_TOLERANCE,
                            "Concat non-identity upper violation: k.y={combined} > ub={crown_upper}, \
                             k=[{k0},{k1},{k2},{k3}], xa=[{xa0},{xa1}], xb=[{xb0},{xb1}]"
                        );
                    }
                }
            }
        }
    }

    /// ConcatLayer CROWN backward with non-identity incoming and non-zero bias.
    ///
    /// Tests the bias-halving directed rounding path: when incoming bounds have
    /// non-zero bias, the bias is split as lower_b/2 (rounded down) and
    /// upper_b/2 (rounded up) between branches. Verifies the split remains sound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_concat_crown_nonidentity_with_bias(
        la0 in -3.0f32..3.0, da0 in 0.01f32..2.0,
        la1 in -3.0f32..3.0, da1 in 0.01f32..2.0,
        lb0 in -3.0f32..3.0, db0 in 0.01f32..2.0,
        lb1 in -3.0f32..3.0, db1 in 0.01f32..2.0,
        bias_lower in -5.0f32..5.0,
        bias_upper in -5.0f32..5.0,
    ) {
        // Ensure bias_upper >= bias_lower for well-formed bounds
        let (bl, bu) = if bias_lower <= bias_upper {
            (bias_lower, bias_upper)
        } else {
            (bias_upper, bias_lower)
        };

        let ua0 = (la0 + da0).min(3.0);
        let ua1 = (la1 + da1).min(3.0);
        let ub0 = (lb0 + db0).min(3.0);
        let ub1 = (lb1 + db1).min(3.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = ConcatLayer::new(0);

        // Identity coefficients but non-zero bias (simulates accumulated bias
        // from upstream CROWN backward passes)
        let incoming = LinearBounds::new(
            Array2::eye(4),
            Array1::from_elem(4, bl),
            Array2::eye(4),
            Array1::from_elem(4, bu),
        ).unwrap();

        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &incoming,
            &[2],
            &[2],
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (bias) failed: {e}")
            ))?;
        let concrete_a = bounds_a.concretize(&input_a);
        let concrete_b = bounds_b.concretize(&input_b);

        // Sampling soundness: concat(a, b) + bias must be within combined bounds
        let spts = sample_points(0.0, 1.0, 5);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0, xa1, xb0, xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            let crown_lower = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
                            let crown_upper = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
                            // True output is yi + bl (lower) through yi + bu (upper)
                            // since the incoming had bias [bl] for lower, [bu] for upper.
                            // CROWN backward should have propagated this, so:
                            // crown_lower <= yi + bl and crown_upper >= yi + bu
                            prop_assert!(
                                yi + bl >= crown_lower - CONCAT_TOLERANCE,
                                "Concat bias lower violation at {i}: y+bl={} < lb={crown_lower}",
                                yi + bl,
                            );
                            prop_assert!(
                                yi + bu <= crown_upper + CONCAT_TOLERANCE,
                                "Concat bias upper violation at {i}: y+bu={} > ub={crown_upper}",
                                yi + bu,
                            );
                        }
                    }
                }
            }
        }
    }
}
