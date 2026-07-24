// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for BilinearCrownLayer (matmul Q @ K^T).
//!
//! BilinearCrownLayer uses McCormick envelope relaxation for the bilinear
//! operation z_{ij} = sum_l q_{il} * k_{jl} (when transpose_b=true).
//! CROWN backward returns (bounds_a, bounds_b) decomposing the relaxation
//! into linear functions of Q and K respectively.
//!
//! Soundness property: for all (Q, K) within input bounds,
//!   concrete_a.lower + concrete_b.lower <= (Q @ K^T)[flat] <= concrete_a.upper + concrete_b.upper
//!
//! Tests use m=n=k=2 (2x2 matmul) with LinearBounds (flat graph network path).
//! This covers `propagate_linear_binary` (fixed McCormick midpoint) and
//! `propagate_linear_binary_with_alpha` (optimizable interpolation parameters).
//!
//! Part of #3104: Binary ops CROWN proptest coverage.
//!
//! Reference: McCormick (1976), "Computability of global solutions to factorable
//! nonconvex programs". Implementation in bilinear.rs:interpolated_mccormick.

use crate::layers::binary_ops::BilinearCrownLayer;
use crate::LinearBounds;
use ndarray::{Array4, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// Tolerance for McCormick relaxation soundness.
/// McCormick involves products of bounds and directed rounding on f64→f32
/// bias halving, accumulating more error than simple affine ops.
const BILINEAR_TOLERANCE: f32 = 1e-3;

/// Helper: create a 2D BoundedTensor from flat lower/upper arrays.
fn make_bt_2d(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Compute true matmul C = Q @ K^T for 2x2 matrices.
/// Q: [m=2, k=2], K: [n=2, k=2], C: [m=2, n=2] flattened to [4].
fn matmul_2x2_transpose(q: &[f32; 4], k: &[f32; 4]) -> [f32; 4] {
    // Q = [[q[0], q[1]], [q[2], q[3]]]
    // K = [[k[0], k[1]], [k[2], k[3]]]
    // C = Q @ K^T where C[i,j] = sum_l Q[i,l] * K[j,l]
    [
        q[0] * k[0] + q[1] * k[1], // C[0,0]
        q[0] * k[2] + q[1] * k[3], // C[0,1]
        q[2] * k[0] + q[3] * k[1], // C[1,0]
        q[2] * k[2] + q[3] * k[3], // C[1,1]
    ]
}

// =============================================================================
// BILINEAR CROWN BACKWARD SOUNDNESS (FIXED McCORMICK)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// BilinearCrownLayer CROWN backward soundness with identity incoming bounds.
    ///
    /// For C = Q @ K^T (2x2 matmul), McCormick relaxation provides sound linear
    /// bounds. Verifies that concretized CROWN bounds contain the true matmul
    /// output for all sampled (Q, K) within the input box.
    ///
    /// Uses moderate input ranges to avoid MCCORMICK_MAX_MAGNITUDE rejection.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_bilinear_crown_identity(
        // Q bounds: 4 elements [m=2, k=2]
        lq0 in -2.0f32..2.0, dq0 in 0.01f32..1.5,
        lq1 in -2.0f32..2.0, dq1 in 0.01f32..1.5,
        lq2 in -2.0f32..2.0, dq2 in 0.01f32..1.5,
        lq3 in -2.0f32..2.0, dq3 in 0.01f32..1.5,
        // K bounds: 4 elements [n=2, k=2]
        lk0 in -2.0f32..2.0, dk0 in 0.01f32..1.5,
        lk1 in -2.0f32..2.0, dk1 in 0.01f32..1.5,
        lk2 in -2.0f32..2.0, dk2 in 0.01f32..1.5,
        lk3 in -2.0f32..2.0, dk3 in 0.01f32..1.5,
    ) {
        let uq = [
            (lq0 + dq0).min(2.0), (lq1 + dq1).min(2.0),
            (lq2 + dq2).min(2.0), (lq3 + dq3).min(2.0),
        ];
        let lq = [lq0, lq1, lq2, lq3];

        let uk = [
            (lk0 + dk0).min(2.0), (lk1 + dk1).min(2.0),
            (lk2 + dk2).min(2.0), (lk3 + dk3).min(2.0),
        ];
        let lk = [lk0, lk1, lk2, lk3];

        let input_q = make_bt_2d(&lq, &uq, &[2, 2]); // [m=2, k=2]
        let input_k = make_bt_2d(&lk, &uk, &[2, 2]); // [n=2, k=2]

        let layer = BilinearCrownLayer::new(true, None); // transpose_b=true, no scale

        // CROWN backward with identity incoming (output size = m*n = 4)
        let identity = LinearBounds::identity(4);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity, &input_q, &input_k,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary failed: {e}")
            ))?;

        let concrete_a = bounds_a.concretize(&input_q);
        let concrete_b = bounds_b.concretize(&input_k);

        // Combined CROWN bounds
        let mut crown_lower = [0.0f32; 4];
        let mut crown_upper = [0.0f32; 4];
        for i in 0..4 {
            crown_lower[i] = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            crown_upper[i] = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
        }

        // Sampling soundness: true matmul Q @ K^T must lie within CROWN bounds
        let spts = sample_points(0.0, 1.0, 5);
        for &tq0 in &spts {
            for &tq1 in &spts {
                for &tk0 in &spts {
                    for &tk1 in &spts {
                        // Sample Q and K values within bounds
                        let q = [
                            lq[0] + tq0 * (uq[0] - lq[0]),
                            lq[1] + tq0 * (uq[1] - lq[1]),
                            lq[2] + tq1 * (uq[2] - lq[2]),
                            lq[3] + tq1 * (uq[3] - lq[3]),
                        ];
                        let k = [
                            lk[0] + tk0 * (uk[0] - lk[0]),
                            lk[1] + tk0 * (uk[1] - lk[1]),
                            lk[2] + tk1 * (uk[2] - lk[2]),
                            lk[3] + tk1 * (uk[3] - lk[3]),
                        ];

                        let c = matmul_2x2_transpose(&q, &k);

                        for (i, &ci) in c.iter().enumerate() {
                            prop_assert!(
                                ci >= crown_lower[i] - BILINEAR_TOLERANCE,
                                "Bilinear lower violation at C[{}]: true={ci} < lb={}, \
                                 Q=[{:.3},{:.3},{:.3},{:.3}], K=[{:.3},{:.3},{:.3},{:.3}]",
                                i, crown_lower[i],
                                q[0], q[1], q[2], q[3],
                                k[0], k[1], k[2], k[3],
                            );
                            prop_assert!(
                                ci <= crown_upper[i] + BILINEAR_TOLERANCE,
                                "Bilinear upper violation at C[{}]: true={ci} > ub={}, \
                                 Q=[{:.3},{:.3},{:.3},{:.3}], K=[{:.3},{:.3},{:.3},{:.3}]",
                                i, crown_upper[i],
                                q[0], q[1], q[2], q[3],
                                k[0], k[1], k[2], k[3],
                            );
                        }
                    }
                }
            }
        }
    }

    /// BilinearCrownLayer CROWN backward soundness with attention scale.
    ///
    /// Tests C = (Q @ K^T) * scale where scale = 1/sqrt(d_k) = 1/sqrt(2) ≈ 0.707.
    /// The scale factor multiplies all McCormick envelope terms and must maintain
    /// soundness through the directed rounding path.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_bilinear_crown_with_scale(
        lq0 in -2.0f32..2.0, dq0 in 0.01f32..1.5,
        lq1 in -2.0f32..2.0, dq1 in 0.01f32..1.5,
        lq2 in -2.0f32..2.0, dq2 in 0.01f32..1.5,
        lq3 in -2.0f32..2.0, dq3 in 0.01f32..1.5,
        lk0 in -2.0f32..2.0, dk0 in 0.01f32..1.5,
        lk1 in -2.0f32..2.0, dk1 in 0.01f32..1.5,
        lk2 in -2.0f32..2.0, dk2 in 0.01f32..1.5,
        lk3 in -2.0f32..2.0, dk3 in 0.01f32..1.5,
    ) {
        let uq = [
            (lq0 + dq0).min(2.0), (lq1 + dq1).min(2.0),
            (lq2 + dq2).min(2.0), (lq3 + dq3).min(2.0),
        ];
        let lq = [lq0, lq1, lq2, lq3];
        let uk = [
            (lk0 + dk0).min(2.0), (lk1 + dk1).min(2.0),
            (lk2 + dk2).min(2.0), (lk3 + dk3).min(2.0),
        ];
        let lk = [lk0, lk1, lk2, lk3];

        let input_q = make_bt_2d(&lq, &uq, &[2, 2]);
        let input_k = make_bt_2d(&lk, &uk, &[2, 2]);

        let scale = 1.0 / (2.0_f32).sqrt(); // 1/sqrt(d_k) for d_k=2
        let layer = BilinearCrownLayer::new(true, Some(scale));

        let identity = LinearBounds::identity(4);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity, &input_q, &input_k,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (scaled) failed: {e}")
            ))?;

        let concrete_a = bounds_a.concretize(&input_q);
        let concrete_b = bounds_b.concretize(&input_k);

        let mut crown_lower = [0.0f32; 4];
        let mut crown_upper = [0.0f32; 4];
        for i in 0..4 {
            crown_lower[i] = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            crown_upper[i] = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
        }

        // True: (Q @ K^T) * scale
        let spts = sample_points(0.0, 1.0, 5);
        for &tq0 in &spts {
            for &tq1 in &spts {
                for &tk0 in &spts {
                    for &tk1 in &spts {
                        let q = [
                            lq[0] + tq0 * (uq[0] - lq[0]),
                            lq[1] + tq0 * (uq[1] - lq[1]),
                            lq[2] + tq1 * (uq[2] - lq[2]),
                            lq[3] + tq1 * (uq[3] - lq[3]),
                        ];
                        let k = [
                            lk[0] + tk0 * (uk[0] - lk[0]),
                            lk[1] + tk0 * (uk[1] - lk[1]),
                            lk[2] + tk1 * (uk[2] - lk[2]),
                            lk[3] + tk1 * (uk[3] - lk[3]),
                        ];

                        let raw = matmul_2x2_transpose(&q, &k);
                        let c: [f32; 4] = [
                            raw[0] * scale, raw[1] * scale,
                            raw[2] * scale, raw[3] * scale,
                        ];

                        for (i, &ci) in c.iter().enumerate() {
                            prop_assert!(
                                ci >= crown_lower[i] - BILINEAR_TOLERANCE,
                                "Scaled bilinear lower violation at C[{}]: true={ci} < lb={}",
                                i, crown_lower[i],
                            );
                            prop_assert!(
                                ci <= crown_upper[i] + BILINEAR_TOLERANCE,
                                "Scaled bilinear upper violation at C[{}]: true={ci} > ub={}",
                                i, crown_upper[i],
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// BILINEAR CROWN WITH ALPHA (INTERPOLATED McCORMICK)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// BilinearCrownLayer alpha-parameterized CROWN backward soundness.
    ///
    /// Tests `propagate_linear_binary_with_alpha` with random alpha values
    /// in [0, 1]. The interpolated McCormick relaxation must remain sound
    /// regardless of alpha choices — only tightness varies.
    ///
    /// Alpha shape: [4, m=2, n=2, k=2] — four interpolation parameters per
    /// (i,j,l) triple controlling McCormick plane selection for lower/upper
    /// bound directions.
    ///
    /// Reference: auto_LiRPA operators/bivariate.py:MulHelper.interpolated_relaxation
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_bilinear_crown_with_alpha(
        lq0 in -2.0f32..2.0, dq0 in 0.01f32..1.5,
        lq1 in -2.0f32..2.0, dq1 in 0.01f32..1.5,
        lq2 in -2.0f32..2.0, dq2 in 0.01f32..1.5,
        lq3 in -2.0f32..2.0, dq3 in 0.01f32..1.5,
        lk0 in -2.0f32..2.0, dk0 in 0.01f32..1.5,
        lk1 in -2.0f32..2.0, dk1 in 0.01f32..1.5,
        lk2 in -2.0f32..2.0, dk2 in 0.01f32..1.5,
        lk3 in -2.0f32..2.0, dk3 in 0.01f32..1.5,
        // Random alpha values for 4 channels, each in [0, 1]
        a0 in 0.0f32..1.0, a1 in 0.0f32..1.0,
        a2 in 0.0f32..1.0, a3 in 0.0f32..1.0,
    ) {
        let uq = [
            (lq0 + dq0).min(2.0), (lq1 + dq1).min(2.0),
            (lq2 + dq2).min(2.0), (lq3 + dq3).min(2.0),
        ];
        let lq = [lq0, lq1, lq2, lq3];
        let uk = [
            (lk0 + dk0).min(2.0), (lk1 + dk1).min(2.0),
            (lk2 + dk2).min(2.0), (lk3 + dk3).min(2.0),
        ];
        let lk = [lk0, lk1, lk2, lk3];

        let input_q = make_bt_2d(&lq, &uq, &[2, 2]);
        let input_k = make_bt_2d(&lk, &uk, &[2, 2]);

        // Build alpha array [4, m=2, n=2, k=2] with uniform values per channel
        // (proptest doesn't easily generate 32 independent params, so use
        // 4 channel-uniform values which still exercise all plane combinations)
        let mut alphas = Array4::<f32>::zeros((4, 2, 2, 2));
        alphas.slice_mut(ndarray::s![0, .., .., ..]).fill(a0); // r_l for lower
        alphas.slice_mut(ndarray::s![1, .., .., ..]).fill(a1); // r_l for upper
        alphas.slice_mut(ndarray::s![2, .., .., ..]).fill(a2); // r_u for lower
        alphas.slice_mut(ndarray::s![3, .., .., ..]).fill(a3); // r_u for upper

        let layer = BilinearCrownLayer::new(true, None);

        let identity = LinearBounds::identity(4);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary_with_alpha(
            &identity, &input_q, &input_k, Some(&alphas),
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary_with_alpha failed: {e}")
            ))?;

        let concrete_a = bounds_a.concretize(&input_q);
        let concrete_b = bounds_b.concretize(&input_k);

        let mut crown_lower = [0.0f32; 4];
        let mut crown_upper = [0.0f32; 4];
        for i in 0..4 {
            crown_lower[i] = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            crown_upper[i] = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
        }

        // Sampling soundness
        let spts = sample_points(0.0, 1.0, 5);
        for &tq0 in &spts {
            for &tq1 in &spts {
                for &tk0 in &spts {
                    for &tk1 in &spts {
                        let q = [
                            lq[0] + tq0 * (uq[0] - lq[0]),
                            lq[1] + tq0 * (uq[1] - lq[1]),
                            lq[2] + tq1 * (uq[2] - lq[2]),
                            lq[3] + tq1 * (uq[3] - lq[3]),
                        ];
                        let k = [
                            lk[0] + tk0 * (uk[0] - lk[0]),
                            lk[1] + tk0 * (uk[1] - lk[1]),
                            lk[2] + tk1 * (uk[2] - lk[2]),
                            lk[3] + tk1 * (uk[3] - lk[3]),
                        ];

                        let c = matmul_2x2_transpose(&q, &k);

                        for (i, &ci) in c.iter().enumerate() {
                            prop_assert!(
                                ci >= crown_lower[i] - BILINEAR_TOLERANCE,
                                "Alpha bilinear lower violation at C[{}]: \
                                 true={ci} < lb={}, alpha=[{a0},{a1},{a2},{a3}]",
                                i, crown_lower[i],
                            );
                            prop_assert!(
                                ci <= crown_upper[i] + BILINEAR_TOLERANCE,
                                "Alpha bilinear upper violation at C[{}]: \
                                 true={ci} > ub={}, alpha=[{a0},{a1},{a2},{a3}]",
                                i, crown_upper[i],
                            );
                        }
                    }
                }
            }
        }
    }

    /// BilinearCrownLayer alpha=None should match fixed McCormick exactly.
    ///
    /// Verifies that `propagate_linear_binary_with_alpha(None)` produces
    /// identical results to `propagate_linear_binary`. This ensures the
    /// None fallback path is correct.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_bilinear_alpha_none_matches_fixed(
        lq0 in -2.0f32..2.0, dq0 in 0.01f32..1.5,
        lq1 in -2.0f32..2.0, dq1 in 0.01f32..1.5,
        lq2 in -2.0f32..2.0, dq2 in 0.01f32..1.5,
        lq3 in -2.0f32..2.0, dq3 in 0.01f32..1.5,
        lk0 in -2.0f32..2.0, dk0 in 0.01f32..1.5,
        lk1 in -2.0f32..2.0, dk1 in 0.01f32..1.5,
        lk2 in -2.0f32..2.0, dk2 in 0.01f32..1.5,
        lk3 in -2.0f32..2.0, dk3 in 0.01f32..1.5,
    ) {
        let uq = [
            (lq0 + dq0).min(2.0), (lq1 + dq1).min(2.0),
            (lq2 + dq2).min(2.0), (lq3 + dq3).min(2.0),
        ];
        let lq = [lq0, lq1, lq2, lq3];
        let uk = [
            (lk0 + dk0).min(2.0), (lk1 + dk1).min(2.0),
            (lk2 + dk2).min(2.0), (lk3 + dk3).min(2.0),
        ];
        let lk = [lk0, lk1, lk2, lk3];

        let input_q = make_bt_2d(&lq, &uq, &[2, 2]);
        let input_k = make_bt_2d(&lk, &uk, &[2, 2]);

        let layer = BilinearCrownLayer::new(true, None);
        let identity = LinearBounds::identity(4);

        // Fixed McCormick
        let (fixed_a, fixed_b) = layer.propagate_linear_binary(
            &identity, &input_q, &input_k,
        )
            .map_err(|e| TestCaseError::fail(
                format!("fixed propagate_linear_binary failed: {e}")
            ))?;

        // Alpha=None should delegate to fixed path
        let (alpha_a, alpha_b) = layer.propagate_linear_binary_with_alpha(
            &identity, &input_q, &input_k, None,
        )
            .map_err(|e| TestCaseError::fail(
                format!("alpha=None propagate_linear_binary_with_alpha failed: {e}")
            ))?;

        let fixed_ca = fixed_a.concretize(&input_q);
        let fixed_cb = fixed_b.concretize(&input_k);
        let alpha_ca = alpha_a.concretize(&input_q);
        let alpha_cb = alpha_b.concretize(&input_k);

        for i in 0..4 {
            let fixed_lower = fixed_ca.lower()[[i]] + fixed_cb.lower()[[i]];
            let fixed_upper = fixed_ca.upper()[[i]] + fixed_cb.upper()[[i]];
            let alpha_lower = alpha_ca.lower()[[i]] + alpha_cb.lower()[[i]];
            let alpha_upper = alpha_ca.upper()[[i]] + alpha_cb.upper()[[i]];

            // Handle ±inf: equal infinities match (inf - inf = NaN fails abs test)
            let lower_matches = fixed_lower == alpha_lower
                || (fixed_lower - alpha_lower).abs() < 1e-6;
            let upper_matches = fixed_upper == alpha_upper
                || (fixed_upper - alpha_upper).abs() < 1e-6;

            prop_assert!(
                lower_matches,
                "Alpha=None lower mismatch at {i}: fixed={fixed_lower}, alpha={alpha_lower}",
            );
            prop_assert!(
                upper_matches,
                "Alpha=None upper mismatch at {i}: fixed={fixed_upper}, alpha={alpha_upper}",
            );
        }
    }
}

// =============================================================================
// BILINEAR CROWN ZERO-CROSSING
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// BilinearCrownLayer soundness for zero-crossing intervals.
    ///
    /// When both Q and K intervals cross zero, the matmul output can be in
    /// any quadrant. The McCormick envelope must handle all quadrant combinations
    /// correctly. This is the hardest case for McCormick relaxation.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_bilinear_crown_zero_crossing(
        // Force zero-crossing: lower < 0 < upper
        lq0 in -2.0f32..-0.01, rq0 in 0.01f32..2.0,
        lq1 in -2.0f32..-0.01, rq1 in 0.01f32..2.0,
        lq2 in -2.0f32..-0.01, rq2 in 0.01f32..2.0,
        lq3 in -2.0f32..-0.01, rq3 in 0.01f32..2.0,
        lk0 in -2.0f32..-0.01, rk0 in 0.01f32..2.0,
        lk1 in -2.0f32..-0.01, rk1 in 0.01f32..2.0,
        lk2 in -2.0f32..-0.01, rk2 in 0.01f32..2.0,
        lk3 in -2.0f32..-0.01, rk3 in 0.01f32..2.0,
    ) {
        let lq = [lq0, lq1, lq2, lq3];
        let uq = [rq0, rq1, rq2, rq3];
        let lk = [lk0, lk1, lk2, lk3];
        let uk = [rk0, rk1, rk2, rk3];

        let input_q = make_bt_2d(&lq, &uq, &[2, 2]);
        let input_k = make_bt_2d(&lk, &uk, &[2, 2]);

        let layer = BilinearCrownLayer::new(true, None);

        let identity = LinearBounds::identity(4);
        let (bounds_a, bounds_b) = layer.propagate_linear_binary(
            &identity, &input_q, &input_k,
        )
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear_binary (zero-crossing) failed: {e}")
            ))?;

        let concrete_a = bounds_a.concretize(&input_q);
        let concrete_b = bounds_b.concretize(&input_k);

        let mut crown_lower = [0.0f32; 4];
        let mut crown_upper = [0.0f32; 4];
        for i in 0..4 {
            crown_lower[i] = concrete_a.lower()[[i]] + concrete_b.lower()[[i]];
            crown_upper[i] = concrete_a.upper()[[i]] + concrete_b.upper()[[i]];
        }

        // Sample with focus on corners and zero crossings
        let spts = sample_points(0.0, 1.0, 5);
        for &tq in &spts {
            for &tk in &spts {
                let q = [
                    lq[0] + tq * (uq[0] - lq[0]),
                    lq[1] + tq * (uq[1] - lq[1]),
                    lq[2] + tq * (uq[2] - lq[2]),
                    lq[3] + tq * (uq[3] - lq[3]),
                ];
                let k = [
                    lk[0] + tk * (uk[0] - lk[0]),
                    lk[1] + tk * (uk[1] - lk[1]),
                    lk[2] + tk * (uk[2] - lk[2]),
                    lk[3] + tk * (uk[3] - lk[3]),
                ];

                let c = matmul_2x2_transpose(&q, &k);

                for (i, &ci) in c.iter().enumerate() {
                    prop_assert!(
                        ci >= crown_lower[i] - BILINEAR_TOLERANCE,
                        "Zero-crossing bilinear lower violation at C[{}]: true={ci} < lb={}",
                        i, crown_lower[i],
                    );
                    prop_assert!(
                        ci <= crown_upper[i] + BILINEAR_TOLERANCE,
                        "Zero-crossing bilinear upper violation at C[{}]: true={ci} > ub={}",
                        i, crown_upper[i],
                    );
                }
            }
        }
    }
}

// =============================================================================
// BATCHED BILINEAR CROWN WITH ALPHA (BROADCAST McCORMICK PATH)
// =============================================================================
//
// Tests `propagate_linear_batched_binary_with_alpha` which uses the broadcast
// McCormick architecture: BilinearRelaxation + compose_backward_broadcast_bidirectional.
// This is the production path for attention verification (#286).
//
// The flat path proptests above test `propagate_linear_binary_with_alpha` (LinearBounds).
// This section covers the batched path (BatchedLinearBounds) which was identified
// as having ZERO direct proptests (P1 Prover finding, commit 521c153).
//
// Reference: auto_LiRPA operators/bivariate.py (element-wise McCormick coefficients)
// Design: designs/2026-03-04-286-attention-bilinear-alternative.md Approach A+B

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Batched BilinearCrown alpha-parameterized CROWN backward soundness.
    ///
    /// Tests the broadcast McCormick path with random alpha values in [0, 1].
    /// Uses BatchedLinearBounds identity downstream (m*n = 4 output elements).
    /// Concretizes Q and K bounds separately and sums to get combined CROWN bounds.
    /// Verifies that true Q @ K^T lies within bounds for all sampled points.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_batched_bilinear_crown_with_alpha(
        // Q bounds: 4 elements [m=2, k=2]
        lq0 in -2.0f32..2.0, dq0 in 0.01f32..1.5,
        lq1 in -2.0f32..2.0, dq1 in 0.01f32..1.5,
        lq2 in -2.0f32..2.0, dq2 in 0.01f32..1.5,
        lq3 in -2.0f32..2.0, dq3 in 0.01f32..1.5,
        // K bounds: 4 elements [n=2, k=2]
        lk0 in -2.0f32..2.0, dk0 in 0.01f32..1.5,
        lk1 in -2.0f32..2.0, dk1 in 0.01f32..1.5,
        lk2 in -2.0f32..2.0, dk2 in 0.01f32..1.5,
        lk3 in -2.0f32..2.0, dk3 in 0.01f32..1.5,
        // Random alpha values for 4 channels, each in [0, 1]
        a0 in 0.0f32..1.0, a1 in 0.0f32..1.0,
        a2 in 0.0f32..1.0, a3 in 0.0f32..1.0,
    ) {
        use crate::BatchedLinearBounds;
        use ndarray::{Array2, IxDyn as IxDynAlias};

        let uq = [
            (lq0 + dq0).min(2.0), (lq1 + dq1).min(2.0),
            (lq2 + dq2).min(2.0), (lq3 + dq3).min(2.0),
        ];
        let lq = [lq0, lq1, lq2, lq3];

        let uk = [
            (lk0 + dk0).min(2.0), (lk1 + dk1).min(2.0),
            (lk2 + dk2).min(2.0), (lk3 + dk3).min(2.0),
        ];
        let lk = [lk0, lk1, lk2, lk3];

        let input_q = make_bt_2d(&lq, &uq, &[2, 2]); // [m=2, k=2]
        let input_k = make_bt_2d(&lk, &uk, &[2, 2]); // [n=2, k=2]

        // Build alpha array [4, m=2, n=2, k=2] with uniform values per channel
        let mut alphas = Array4::<f32>::zeros((4, 2, 2, 2));
        alphas.slice_mut(ndarray::s![0, .., .., ..]).fill(a0);
        alphas.slice_mut(ndarray::s![1, .., .., ..]).fill(a1);
        alphas.slice_mut(ndarray::s![2, .., .., ..]).fill(a2);
        alphas.slice_mut(ndarray::s![3, .., .., ..]).fill(a3);

        let layer = BilinearCrownLayer::new(true, None); // transpose_b=true

        // BatchedLinearBounds identity: [z_size, z_size] eye matrix
        // z_size = m * n = 4, input_shape = [m, n] = [2, 2]
        let z_size = 4;
        let downstream = BatchedLinearBounds::new(
            Array2::eye(z_size).into_dyn(),
            ArrayD::zeros(IxDynAlias(&[z_size])),
            Array2::eye(z_size).into_dyn(),
            ArrayD::zeros(IxDynAlias(&[z_size])),
            vec![2, 2],
            vec![2, 2],
        ).map_err(|e| TestCaseError::fail(
            format!("BatchedLinearBounds identity: {e}")
        ))?;

        let (bounds_q, bounds_k) = layer.propagate_linear_batched_binary_with_alpha(
            &downstream, &input_q, &input_k, Some(&alphas),
        ).map_err(|e| TestCaseError::fail(
            format!("propagate_linear_batched_binary_with_alpha failed: {e}")
        ))?;

        // Concretize Q and K bounds separately
        let concrete_q = bounds_q.concretize(&input_q)
            .map_err(|e| TestCaseError::fail(
                format!("concretize Q failed: {e}")
            ))?;
        let concrete_k = bounds_k.concretize(&input_k)
            .map_err(|e| TestCaseError::fail(
                format!("concretize K failed: {e}")
            ))?;

        // Combined CROWN bounds: Q contribution + K contribution
        let q_lower: Vec<f32> = concrete_q.lower().iter().copied().collect();
        let q_upper: Vec<f32> = concrete_q.upper().iter().copied().collect();
        let k_lower: Vec<f32> = concrete_k.lower().iter().copied().collect();
        let k_upper: Vec<f32> = concrete_k.upper().iter().copied().collect();

        let mut crown_lower = [0.0f32; 4];
        let mut crown_upper = [0.0f32; 4];
        for i in 0..4 {
            crown_lower[i] = q_lower[i] + k_lower[i];
            crown_upper[i] = q_upper[i] + k_upper[i];
        }

        // Sampling soundness: true matmul Q @ K^T must lie within CROWN bounds
        let spts = sample_points(0.0, 1.0, 5);
        for &tq0 in &spts {
            for &tq1 in &spts {
                for &tk0 in &spts {
                    for &tk1 in &spts {
                        let q = [
                            lq[0] + tq0 * (uq[0] - lq[0]),
                            lq[1] + tq0 * (uq[1] - lq[1]),
                            lq[2] + tq1 * (uq[2] - lq[2]),
                            lq[3] + tq1 * (uq[3] - lq[3]),
                        ];
                        let k = [
                            lk[0] + tk0 * (uk[0] - lk[0]),
                            lk[1] + tk0 * (uk[1] - lk[1]),
                            lk[2] + tk1 * (uk[2] - lk[2]),
                            lk[3] + tk1 * (uk[3] - lk[3]),
                        ];

                        let c = matmul_2x2_transpose(&q, &k);

                        for (i, &ci) in c.iter().enumerate() {
                            prop_assert!(
                                ci >= crown_lower[i] - BILINEAR_TOLERANCE,
                                "Batched alpha bilinear lower violation at C[{}]: \
                                 true={ci} < lb={}, alpha=[{a0},{a1},{a2},{a3}]",
                                i, crown_lower[i],
                            );
                            prop_assert!(
                                ci <= crown_upper[i] + BILINEAR_TOLERANCE,
                                "Batched alpha bilinear upper violation at C[{}]: \
                                 true={ci} > ub={}, alpha=[{a0},{a1},{a2},{a3}]",
                                i, crown_upper[i],
                            );
                        }
                    }
                }
            }
        }
    }
}
