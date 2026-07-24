// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP soundness proptests for binary operations that lack CROWN backward:
//! DivLayer, MinBinaryLayer, MaxBinaryLayer.
//!
//! These ops only have IBP (interval bound propagation), no linear relaxation.
//! Soundness property: for all (x_a, x_b) in input bounds,
//!   f(x_a, x_b) lies within IBP output bounds.
//!
//! Split from crown_binary_ops.rs to keep file sizes under 1000 lines.
//! Part of #3370.

use crate::layers::binary_ops::{DivLayer, MaxBinaryLayer, MinBinaryLayer};
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Tolerance for Div IBP soundness. DivLayer uses directed rounding, so bounds
/// should be tight to 1 ULP. Small buffer for reference eval rounding.
const DIV_IBP_TOLERANCE: f32 = 1e-6;

/// Tolerance for Min/Max IBP soundness. These are exact (element-wise min/max
/// of bound endpoints), so bounds are tight. Small buffer for reference eval.
const MINMAX_IBP_TOLERANCE: f32 = 1e-6;

// =============================================================================
// DIV BINARY IBP SOUNDNESS
// =============================================================================
//
// DivLayer computes C = A / B where B > 0. IBP bounds use sign-dependent
// monotonicity of f(b) = a/b with directed rounding (next_down_f32/next_up_f32).
// No CROWN backward support exists for DivLayer.
//
// Part of #3370.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// DivLayer IBP soundness: for any (a, b) in input bounds where b > 0,
    /// a/b lies within computed output bounds.
    ///
    /// Tests the three sign cases for A: A_l >= 0, A_u <= 0, and mixed sign.
    /// B is always strictly positive (DivLayer requirement).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_div_binary_ibp(
        la0 in -5.0f32..5.0, da0 in 0.01f32..3.0,
        la1 in -5.0f32..5.0, da1 in 0.01f32..3.0,
        lb0 in 0.1f32..5.0, db0 in 0.01f32..3.0,
        lb1 in 0.1f32..5.0, db1 in 0.01f32..3.0,
    ) {
        let ua0 = (la0 + da0).min(5.0);
        let ua1 = (la1 + da1).min(5.0);
        let ub0 = (lb0 + db0).min(5.0);
        let ub1 = (lb1 + db1).min(5.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = DivLayer;
        let output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // Sampling soundness: a/b must lie within IBP bounds
        let spts = sample_points(0.0, 1.0, 7);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        // Clamp: f32 interpolation can overshoot bounds
                        let xa0 = (la0 + ta0 * (ua0 - la0)).clamp(la0, ua0);
                        let xa1 = (la1 + ta1 * (ua1 - la1)).clamp(la1, ua1);
                        let xb0 = (lb0 + tb0 * (ub0 - lb0)).clamp(lb0, ub0);
                        let xb1 = (lb1 + tb1 * (ub1 - lb1)).clamp(lb1, ub1);
                        let y = [xa0 / xb0, xa1 / xb1];

                        for (i, &yi) in y.iter().enumerate() {
                            prop_assert!(
                                yi >= output.lower()[[i]] - DIV_IBP_TOLERANCE,
                                "Div IBP lower violation at {i}: a/b={yi} < lb={}, \
                                 a={}, b={}, bounds_a=[{},{}], bounds_b=[{},{}]",
                                output.lower()[[i]],
                                if i == 0 { xa0 } else { xa1 },
                                if i == 0 { xb0 } else { xb1 },
                                if i == 0 { la0 } else { la1 },
                                if i == 0 { ua0 } else { ua1 },
                                if i == 0 { lb0 } else { lb1 },
                                if i == 0 { ub0 } else { ub1 },
                            );
                            prop_assert!(
                                yi <= output.upper()[[i]] + DIV_IBP_TOLERANCE,
                                "Div IBP upper violation at {i}: a/b={yi} > ub={}, \
                                 a={}, b={}",
                                output.upper()[[i]],
                                if i == 0 { xa0 } else { xa1 },
                                if i == 0 { xb0 } else { xb1 },
                            );
                        }
                    }
                }
            }
        }
    }

    /// DivLayer IBP soundness with zero-crossing numerator.
    ///
    /// When A crosses zero (A_l < 0 < A_u), the mixed-sign case uses
    /// A_l/B_l for lower and A_u/B_l for upper. This tests that specific path.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_div_binary_ibp_zero_crossing_numerator(
        la0 in -5.0f32..-0.01, ra0 in 0.01f32..5.0,
        la1 in -5.0f32..-0.01, ra1 in 0.01f32..5.0,
        lb0 in 0.1f32..5.0, db0 in 0.01f32..3.0,
        lb1 in 0.1f32..5.0, db1 in 0.01f32..3.0,
    ) {
        // Force zero-crossing: A_l < 0 < A_u
        let ua0 = ra0;
        let ua1 = ra1;
        let ub0 = (lb0 + db0).min(5.0);
        let ub1 = (lb1 + db1).min(5.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = DivLayer;
        let output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        let spts = sample_points(0.0, 1.0, 7);
        for &ta in &spts {
            for &tb in &spts {
                // Clamp interpolated samples to input bounds: f32 arithmetic
                // in `la + ta * (ua - la)` can overshoot ua when the interval
                // is large (catastrophic cancellation).
                let xa0 = (la0 + ta * (ua0 - la0)).clamp(la0, ua0);
                let xa1 = (la1 + ta * (ua1 - la1)).clamp(la1, ua1);
                let xb0 = (lb0 + tb * (ub0 - lb0)).clamp(lb0, ub0);
                let xb1 = (lb1 + tb * (ub1 - lb1)).clamp(lb1, ub1);
                let y = [xa0 / xb0, xa1 / xb1];

                for (i, &yi) in y.iter().enumerate() {
                    prop_assert!(
                        yi >= output.lower()[[i]] - DIV_IBP_TOLERANCE,
                        "Div zero-crossing lower violation at {i}: y={yi} < lb={}",
                        output.lower()[[i]]
                    );
                    prop_assert!(
                        yi <= output.upper()[[i]] + DIV_IBP_TOLERANCE,
                        "Div zero-crossing upper violation at {i}: y={yi} > ub={}",
                        output.upper()[[i]]
                    );
                }
            }
        }
    }
}

// =============================================================================
// MIN BINARY IBP SOUNDNESS
// =============================================================================
//
// MinBinaryLayer computes C = min(A, B) element-wise.
// IBP: C_lower = min(A_l, B_l), C_upper = min(A_u, B_u).
// No CROWN backward support.
//
// Part of #3370.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// MinBinaryLayer IBP soundness: for any (a, b) in input bounds,
    /// min(a, b) lies within computed output bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_min_binary_ibp(
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

        let layer = MinBinaryLayer;
        let output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // Sampling soundness: min(a, b) must lie within IBP bounds
        let spts = sample_points(0.0, 1.0, 7);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0.min(xb0), xa1.min(xb1)];

                        for (i, &yi) in y.iter().enumerate() {
                            prop_assert!(
                                yi >= output.lower()[[i]] - MINMAX_IBP_TOLERANCE,
                                "MinBinary IBP lower violation at {i}: min(a,b)={yi} < lb={}",
                                output.lower()[[i]]
                            );
                            prop_assert!(
                                yi <= output.upper()[[i]] + MINMAX_IBP_TOLERANCE,
                                "MinBinary IBP upper violation at {i}: min(a,b)={yi} > ub={}",
                                output.upper()[[i]]
                            );
                        }
                    }
                }
            }
        }
    }

    /// MinBinaryLayer IBP soundness with negative inputs.
    ///
    /// Exercises the case where both A and B have negative bounds,
    /// common in residual connections after subtraction.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_min_binary_ibp_negative(
        la0 in -10.0f32..-0.1, da0 in 0.01f32..5.0,
        la1 in -10.0f32..-0.1, da1 in 0.01f32..5.0,
        lb0 in -10.0f32..-0.1, db0 in 0.01f32..5.0,
        lb1 in -10.0f32..-0.1, db1 in 0.01f32..5.0,
    ) {
        let ua0 = (la0 + da0).min(0.0);
        let ua1 = (la1 + da1).min(0.0);
        let ub0 = (lb0 + db0).min(0.0);
        let ub1 = (lb1 + db1).min(0.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MinBinaryLayer;
        let output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        let spts = sample_points(0.0, 1.0, 7);
        for &ta in &spts {
            for &tb in &spts {
                let xa0 = la0 + ta * (ua0 - la0);
                let xa1 = la1 + ta * (ua1 - la1);
                let xb0 = lb0 + tb * (ub0 - lb0);
                let xb1 = lb1 + tb * (ub1 - lb1);
                let y = [xa0.min(xb0), xa1.min(xb1)];

                for (i, &yi) in y.iter().enumerate() {
                    prop_assert!(
                        yi >= output.lower()[[i]] - MINMAX_IBP_TOLERANCE,
                        "MinBinary negative lower violation at {i}: y={yi} < lb={}",
                        output.lower()[[i]]
                    );
                    prop_assert!(
                        yi <= output.upper()[[i]] + MINMAX_IBP_TOLERANCE,
                        "MinBinary negative upper violation at {i}: y={yi} > ub={}",
                        output.upper()[[i]]
                    );
                }
            }
        }
    }
}

// =============================================================================
// MAX BINARY IBP SOUNDNESS
// =============================================================================
//
// MaxBinaryLayer computes C = max(A, B) element-wise.
// IBP: C_lower = max(A_l, B_l), C_upper = max(A_u, B_u).
// No CROWN backward support.
//
// Part of #3370.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// MaxBinaryLayer IBP soundness: for any (a, b) in input bounds,
    /// max(a, b) lies within computed output bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_max_binary_ibp(
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

        let layer = MaxBinaryLayer;
        let output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        // Sampling soundness: max(a, b) must lie within IBP bounds
        let spts = sample_points(0.0, 1.0, 7);
        for &ta0 in &spts {
            for &ta1 in &spts {
                for &tb0 in &spts {
                    for &tb1 in &spts {
                        let xa0 = la0 + ta0 * (ua0 - la0);
                        let xa1 = la1 + ta1 * (ua1 - la1);
                        let xb0 = lb0 + tb0 * (ub0 - lb0);
                        let xb1 = lb1 + tb1 * (ub1 - lb1);
                        let y = [xa0.max(xb0), xa1.max(xb1)];

                        for (i, &yi) in y.iter().enumerate() {
                            prop_assert!(
                                yi >= output.lower()[[i]] - MINMAX_IBP_TOLERANCE,
                                "MaxBinary IBP lower violation at {i}: max(a,b)={yi} < lb={}",
                                output.lower()[[i]]
                            );
                            prop_assert!(
                                yi <= output.upper()[[i]] + MINMAX_IBP_TOLERANCE,
                                "MaxBinary IBP upper violation at {i}: max(a,b)={yi} > ub={}",
                                output.upper()[[i]]
                            );
                        }
                    }
                }
            }
        }
    }

    /// MaxBinaryLayer IBP soundness with negative inputs.
    ///
    /// Exercises the case where both A and B have negative bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_max_binary_ibp_negative(
        la0 in -10.0f32..-0.1, da0 in 0.01f32..5.0,
        la1 in -10.0f32..-0.1, da1 in 0.01f32..5.0,
        lb0 in -10.0f32..-0.1, db0 in 0.01f32..5.0,
        lb1 in -10.0f32..-0.1, db1 in 0.01f32..5.0,
    ) {
        let ua0 = (la0 + da0).min(0.0);
        let ua1 = (la1 + da1).min(0.0);
        let ub0 = (lb0 + db0).min(0.0);
        let ub1 = (lb1 + db1).min(0.0);

        let input_a = make_bt(&[la0, la1], &[ua0, ua1]);
        let input_b = make_bt(&[lb0, lb1], &[ub0, ub1]);

        let layer = MaxBinaryLayer;
        let output = layer.propagate_ibp_binary(&input_a, &input_b)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp_binary failed: {e}")
            ))?;

        let spts = sample_points(0.0, 1.0, 7);
        for &ta in &spts {
            for &tb in &spts {
                let xa0 = la0 + ta * (ua0 - la0);
                let xa1 = la1 + ta * (ua1 - la1);
                let xb0 = lb0 + tb * (ub0 - lb0);
                let xb1 = lb1 + tb * (ub1 - lb1);
                let y = [xa0.max(xb0), xa1.max(xb1)];

                for (i, &yi) in y.iter().enumerate() {
                    prop_assert!(
                        yi >= output.lower()[[i]] - MINMAX_IBP_TOLERANCE,
                        "MaxBinary negative lower violation at {i}: y={yi} < lb={}",
                        output.lower()[[i]]
                    );
                    prop_assert!(
                        yi <= output.upper()[[i]] + MINMAX_IBP_TOLERANCE,
                        "MaxBinary negative upper violation at {i}: y={yi} > ub={}",
                        output.upper()[[i]]
                    );
                }
            }
        }
    }
}
