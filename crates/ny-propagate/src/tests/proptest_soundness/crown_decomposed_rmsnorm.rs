// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for decomposed RmsNorm CROWN backward.
//!
//! Tests `decomposed_rms_norm_crown_backward` (crown_block_wise.rs) which propagates
//! CROWN backward through a decomposed RmsNorm chain:
//!   x → x² → mean(x²) → sqrt(mean(x²)+eps) → 1/rms → x*inv_rms → γ·norm
//!
//! Soundness property: for any concrete x in [x_l, x_u], the true RmsNorm
//! output must fall within the CROWN-concretized bounds.
//!
//! Part of #3387 (RmsNorm decomposed CROWN backward).

use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use crate::bounds::BatchedLinearBounds;
use crate::network::decomposed_rms_norm_crown_backward;

use super::{rms_norm, sample_points};

/// Tolerance for decomposed RmsNorm CROWN soundness.
///
/// Same rationale as LayerNorm DECOMPOSED_NORM_TOLERANCE (1e-2):
/// McCormick bilinear approximation + 3-level composition (Reciprocal → Sqrt → Square)
/// + f64→f32 directed rounding at boundaries + fan-out accumulation at x.
const DECOMPOSED_RMSNORM_TOLERANCE: f32 = 1e-2;

/// Core soundness verification: given CROWN result from decomposed RmsNorm
/// backward, verify that for sampled concrete inputs the true RmsNorm output
/// falls within the concretized bounds.
fn verify_decomposed_rmsnorm_soundness(
    ny: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    num_interior_samples: usize,
) -> Result<(), TestCaseError> {
    let shape = x_ibp.shape();
    let n = *shape.last().unwrap_or(&0);
    if n == 0 {
        return Err(TestCaseError::fail(
            "decomposed RmsNorm soundness oracle requires a nonempty normalized axis",
        ));
    }

    // Identity incoming A (output = input passthrough)
    let identity = BatchedLinearBounds::identity(shape)
        .map_err(|e| TestCaseError::fail(format!("identity creation failed: {e}")))?;

    // Run decomposed RmsNorm CROWN backward
    let crown_result =
        decomposed_rms_norm_crown_backward(&identity, ny, eps, x_ibp).map_err(|e| {
            TestCaseError::fail(format!("decomposed_rms_norm_crown_backward failed: {e}"))
        })?;

    // Concretize: compute [lb, ub] by optimizing A@x+b over x in [x_l, x_u]
    let concretized = crown_result
        .bounds
        .concretize_sound(x_ibp)
        .map_err(|e| TestCaseError::fail(format!("concretize_sound failed: {e}")))?;

    let lb: Vec<f32> = concretized.lower().iter().copied().collect();
    let ub: Vec<f32> = concretized.upper().iter().copied().collect();

    // Sample concrete inputs and verify RmsNorm output is within bounds
    let x_lower: Vec<f32> = x_ibp.lower().iter().copied().collect();
    let x_upper: Vec<f32> = x_ibp.upper().iter().copied().collect();

    let per_dim_samples: Vec<Vec<f32>> = (0..n)
        .map(|i| sample_points(x_lower[i], x_upper[i], num_interior_samples))
        .collect();

    let mut test_points: Vec<Vec<f32>> = Vec::new();

    // All-lower and all-upper corners
    test_points.push(x_lower.clone());
    test_points.push(x_upper.clone());

    // Per-dimension extremes at midpoint
    let midpoints: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| l * 0.5 + u * 0.5)
        .collect();
    for i in 0..n {
        let mut pt_lo = midpoints.clone();
        pt_lo[i] = x_lower[i];
        test_points.push(pt_lo);

        let mut pt_hi = midpoints.clone();
        pt_hi[i] = x_upper[i];
        test_points.push(pt_hi);
    }

    // Interior samples
    for sample_idx in 0..num_interior_samples {
        let point: Vec<f32> = (0..n)
            .map(|dim| {
                let idx = (sample_idx + dim * 7) % per_dim_samples[dim].len();
                per_dim_samples[dim][idx]
            })
            .collect();
        test_points.push(point);
    }

    for point in &test_points {
        let x_arr = Array1::from_vec(point.clone());
        let true_output = rms_norm(&x_arr, ny, eps);

        for i in 0..n {
            let fx = true_output[i];
            let scale_tol = DECOMPOSED_RMSNORM_TOLERANCE * fx.abs().max(1.0);

            prop_assert!(
                lb[i] <= fx + scale_tol,
                "Decomposed RmsNorm CROWN lower bound violated at dim {i}: \
                 lb={} > f(x)={fx} (tol={scale_tol}, ny={}, eps={eps})",
                lb[i],
                ny[i],
            );
            prop_assert!(
                ub[i] + scale_tol >= fx,
                "Decomposed RmsNorm CROWN upper bound violated at dim {i}: \
                 ub={} < f(x)={fx} (tol={scale_tol}, ny={}, eps={eps})",
                ub[i],
                ny[i],
            );
        }
    }

    Ok(())
}

// =============================================================================
// PROPTEST: DECOMPOSED RMSNORM CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Decomposed RmsNorm CROWN soundness with tight perturbation (n=3).
    ///
    /// Part of #3387.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_rmsnorm_crown_identity_tight_n3(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw in 0.01f32..0.2,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        verify_decomposed_rmsnorm_soundness(&ny, eps, &x_ibp, 20)?;
    }

    /// Decomposed RmsNorm CROWN with wider perturbation (n=3, hw up to 0.5).
    ///
    /// Part of #3387.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_rmsnorm_crown_wide_perturbation_n3(
        c0 in -1.0f32..1.0,
        c1 in -1.0f32..1.0,
        c2 in -1.0f32..1.0,
        hw in 0.1f32..0.5,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        verify_decomposed_rmsnorm_soundness(&ny, eps, &x_ibp, 20)?;
    }

    /// Decomposed RmsNorm CROWN with larger dimension (n=8).
    ///
    /// Part of #3387.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_rmsnorm_crown_identity_tight_n8(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        c6 in -2.0f32..2.0,
        c7 in -2.0f32..2.0,
        hw in 0.01f32..0.15,
        g_scale in 0.5f32..2.0,
    ) {
        let n = 8;
        let ny = Array1::from_elem(n, g_scale);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        verify_decomposed_rmsnorm_soundness(&ny, eps, &x_ibp, 30)?;
    }

    /// Decomposed RmsNorm CROWN with asymmetric incoming A matrices.
    ///
    /// Tests McCormick plane selection when incoming coefficients are negative.
    ///
    /// Part of #3387.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_rmsnorm_crown_negcoeff_n3(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw in 0.01f32..0.15,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        w0 in -2.0f32..2.0,
        w1 in -2.0f32..2.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        // Asymmetric incoming A: 2 output dims with mixed-sign weights
        let out_dim = 2;
        let a_data_l = vec![w0, 0.0, w1, 0.0, w0, 0.0];
        let a_data_u = vec![w0, 0.0, w1, 0.0, w0, 0.0];
        let b_data_l = vec![0.0_f32; out_dim];
        let b_data_u = vec![0.0_f32; out_dim];

        let incoming = BatchedLinearBounds::new(
            ArrayD::from_shape_vec(IxDyn(&[out_dim, n]), a_data_l).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_dim]), b_data_l).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_dim, n]), a_data_u).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_dim]), b_data_u).unwrap(),
            vec![n],
            vec![out_dim],
        ).unwrap();

        // Reject degenerate all-zero A matrices
        prop_assume!(
            w0.abs() > f32::EPSILON || w1.abs() > f32::EPSILON,
            "Degenerate: both weights ~0, no negative coefficients to test"
        );

        let result = decomposed_rms_norm_crown_backward(&incoming, &ny, eps, &x_ibp)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "decomposed_rms_norm_crown_backward failed unexpectedly: {e}"
                ))
            })?;

        let concretized: BoundedTensor = result.bounds.concretize_sound(&x_ibp)
            .map_err(|e| TestCaseError::fail(
                format!("concretize failed: {e}")
            ))?;

        let lb: Vec<f32> = concretized.lower().iter().copied().collect();
        let ub: Vec<f32> = concretized.upper().iter().copied().collect();

        let x_lower: Vec<f32> = x_ibp.lower().iter().copied().collect();
        let x_upper: Vec<f32> = x_ibp.upper().iter().copied().collect();
        let per_dim_samples: Vec<Vec<f32>> = (0..n)
            .map(|i| sample_points(x_lower[i], x_upper[i], 20))
            .collect();

        let mut test_points: Vec<Vec<f32>> = vec![x_lower.clone(), x_upper.clone()];
        let midpoints: Vec<f32> = x_lower.iter().zip(x_upper.iter())
            .map(|(&l, &u)| l * 0.5 + u * 0.5).collect();
        for i in 0..n {
            let mut pt_lo = midpoints.clone();
            pt_lo[i] = x_lower[i];
            test_points.push(pt_lo);
            let mut pt_hi = midpoints.clone();
            pt_hi[i] = x_upper[i];
            test_points.push(pt_hi);
        }
        for sample_idx in 0..20 {
            let point: Vec<f32> = (0..n)
                .map(|dim| {
                    let idx = (sample_idx + dim * 7) % per_dim_samples[dim].len();
                    per_dim_samples[dim][idx]
                })
                .collect();
            test_points.push(point);
        }

        for point in &test_points {
            let x_arr = Array1::from_vec(point.clone());
            let true_out = rms_norm(&x_arr, &ny, eps);

            for o in 0..out_dim {
                let mut true_val = 0.0_f32;
                for j in 0..n {
                    true_val += incoming.lower_a()[[o, j]] * true_out[j];
                }
                true_val += incoming.lower_b()[o];

                let scale_tol = DECOMPOSED_RMSNORM_TOLERANCE * true_val.abs().max(1.0);

                prop_assert!(
                    lb[o] <= true_val + scale_tol,
                    "Negcoeff lower violated at out_dim {o}: \
                     lb={} > val={true_val} (tol={scale_tol}, w0={w0}, w1={w1})",
                    lb[o],
                );
                prop_assert!(
                    ub[o] + scale_tol >= true_val,
                    "Negcoeff upper violated at out_dim {o}: \
                     ub={} < val={true_val} (tol={scale_tol}, w0={w0}, w1={w1})",
                    ub[o],
                );
            }
        }
    }

    /// Decomposed RmsNorm CROWN with asymmetric incoming A (lower_a != upper_a).
    ///
    /// Tests the general nested CROWN case where outer layers produce different
    /// linear relaxations for lower vs upper bounds. The McCormick plane selection
    /// at lines 1918 and 1946 of crown_block_wise.rs uses w_l and w_u independently,
    /// so this test exercises paths where w_l and w_u have different signs.
    ///
    /// Part of #3387.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_rmsnorm_crown_asymmetric_a_n3(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw in 0.01f32..0.15,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        w_l0 in -2.0f32..2.0,
        w_l1 in -2.0f32..2.0,
        w_u0 in -2.0f32..2.0,
        w_u1 in -2.0f32..2.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        // Reject degenerate all-zero A matrices
        prop_assume!(
            w_l0.abs() > f32::EPSILON || w_l1.abs() > f32::EPSILON
                || w_u0.abs() > f32::EPSILON || w_u1.abs() > f32::EPSILON,
            "Degenerate: all weights ~0"
        );

        // Asymmetric incoming A: lower_a != upper_a (2 output dims × 3 input dims)
        let out_dim = 2;
        let a_data_l = vec![w_l0, 0.0, w_l1, 0.0, w_l0, 0.0];
        let a_data_u = vec![w_u0, 0.0, w_u1, 0.0, w_u0, 0.0];
        let b_data_l = vec![0.0_f32; out_dim];
        let b_data_u = vec![0.0_f32; out_dim];

        let incoming = BatchedLinearBounds::new(
            ArrayD::from_shape_vec(IxDyn(&[out_dim, n]), a_data_l).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_dim]), b_data_l).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_dim, n]), a_data_u).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_dim]), b_data_u).unwrap(),
            vec![n],
            vec![out_dim],
        ).unwrap();

        // Filter out infeasible incoming bounds: at the center point, the
        // lower bound expression must be <= upper bound expression. When
        // lower_a != upper_a, some coefficient combinations are geometrically
        // contradictory (e.g., lower_a=0, upper_a<0 with positive RmsNorm output).
        let center_arr = Array1::from_vec(centers.to_vec());
        let center_out = rms_norm(&center_arr, &ny, eps);
        for o in 0..out_dim {
            let val_l: f32 = (0..n).map(|j| incoming.lower_a()[[o, j]] * center_out[j]).sum::<f32>()
                + incoming.lower_b()[o];
            let val_u: f32 = (0..n).map(|j| incoming.upper_a()[[o, j]] * center_out[j]).sum::<f32>()
                + incoming.upper_b()[o];
            let feasibility_msg = format!(
                "Infeasible incoming bounds at center (dim {o}): lower={val_l} > upper={val_u}"
            );
            prop_assume!(val_l <= val_u + 1e-4, "{}", feasibility_msg);
        }

        let result = decomposed_rms_norm_crown_backward(&incoming, &ny, eps, &x_ibp)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "decomposed_rms_norm_crown_backward failed unexpectedly: {e}"
                ))
            })?;

        let concretized: BoundedTensor = result.bounds.concretize_sound(&x_ibp)
            .map_err(|e| TestCaseError::fail(
                format!("concretize failed: {e}")
            ))?;

        let lb: Vec<f32> = concretized.lower().iter().copied().collect();
        let ub: Vec<f32> = concretized.upper().iter().copied().collect();

        let x_lower: Vec<f32> = x_ibp.lower().iter().copied().collect();
        let x_upper: Vec<f32> = x_ibp.upper().iter().copied().collect();
        let per_dim_samples: Vec<Vec<f32>> = (0..n)
            .map(|i| sample_points(x_lower[i], x_upper[i], 20))
            .collect();

        let mut test_points: Vec<Vec<f32>> = vec![x_lower.clone(), x_upper.clone()];
        let midpoints: Vec<f32> = x_lower.iter().zip(x_upper.iter())
            .map(|(&l, &u)| l * 0.5 + u * 0.5).collect();
        for i in 0..n {
            let mut pt_lo = midpoints.clone();
            pt_lo[i] = x_lower[i];
            test_points.push(pt_lo);
            let mut pt_hi = midpoints.clone();
            pt_hi[i] = x_upper[i];
            test_points.push(pt_hi);
        }
        for sample_idx in 0..20 {
            let point: Vec<f32> = (0..n)
                .map(|dim| {
                    let idx = (sample_idx + dim * 7) % per_dim_samples[dim].len();
                    per_dim_samples[dim][idx]
                })
                .collect();
            test_points.push(point);
        }

        for point in &test_points {
            let x_arr = Array1::from_vec(point.clone());
            let true_out = rms_norm(&x_arr, &ny, eps);

            for o in 0..out_dim {
                // Lower bound check: lb <= incoming.lower_a @ RmsNorm(x) + lower_b
                let mut true_val_lower = 0.0_f32;
                for j in 0..n {
                    true_val_lower += incoming.lower_a()[[o, j]] * true_out[j];
                }
                true_val_lower += incoming.lower_b()[o];

                // Upper bound check: ub >= incoming.upper_a @ RmsNorm(x) + upper_b
                let mut true_val_upper = 0.0_f32;
                for j in 0..n {
                    true_val_upper += incoming.upper_a()[[o, j]] * true_out[j];
                }
                true_val_upper += incoming.upper_b()[o];

                let scale_tol_l = DECOMPOSED_RMSNORM_TOLERANCE * true_val_lower.abs().max(1.0);
                let scale_tol_u = DECOMPOSED_RMSNORM_TOLERANCE * true_val_upper.abs().max(1.0);

                prop_assert!(
                    lb[o] <= true_val_lower + scale_tol_l,
                    "Asymmetric-A lower violated at out_dim {o}: \
                     lb={} > val={true_val_lower} (tol={scale_tol_l}, \
                     w_l0={w_l0}, w_l1={w_l1}, w_u0={w_u0}, w_u1={w_u1})",
                    lb[o],
                );
                prop_assert!(
                    ub[o] + scale_tol_u >= true_val_upper,
                    "Asymmetric-A upper violated at out_dim {o}: \
                     ub={} < val={true_val_upper} (tol={scale_tol_u}, \
                     w_l0={w_l0}, w_l1={w_l1}, w_u0={w_u0}, w_u1={w_u1})",
                    ub[o],
                );
            }
        }
    }
}
