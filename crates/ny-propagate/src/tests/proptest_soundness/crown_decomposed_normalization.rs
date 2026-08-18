// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for decomposed LayerNorm CROWN backward.
//!
//! Tests `decomposed_norm_crown_backward` (crown_block_wise.rs) which propagates
//! CROWN backward through a decomposed normalization chain:
//!   x → mean(x) → d=x-mean → d² → var=mean(d²) → sqrt(var+eps) → 1/std → d*inv_std → γ·norm+β
//!
//! Soundness property: for any concrete x in [x_l, x_u], the true LayerNorm
//! output must fall within the CROWN-concretized bounds.
//!
//! Part of #318 (Path 6b acceptance criterion: proptest soundness).

use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use crate::bounds::BatchedLinearBounds;
use crate::layers::common::BoundPropagation;
use crate::layers::LayerNormLayer;
use crate::network::decomposed_norm_crown_backward;

use super::{layernorm, sample_points};

/// Tolerance for decomposed normalization CROWN soundness.
///
/// Higher than standard FP_TOLERANCE (1e-5) because:
/// - McCormick relaxation introduces bilinear approximation error
/// - 3-level composition (Reciprocal → Sqrt → Square) compounds rounding
/// - f64→f32 directed rounding at decomposition boundaries adds ±1 ULP per step
/// - Fan-out accumulation at d node sums two independently-rounded paths
///
/// The 1e-2 tolerance matches the other normalization CROWN soundness tests
/// in crown_normalization_layernorm.rs.
const DECOMPOSED_NORM_TOLERANCE: f32 = 1e-2;

#[test]
fn test_decomposed_norm_crown_near_constant_interval_stays_within_fused_ibp() {
    let n = 8;
    let ny = Array1::ones(n);
    let beta = Array1::zeros(n);
    let eps = 1e-5_f32;

    // Near-constant intervals drive std_lower toward sqrt(eps), which makes
    // the unclipped d * inv_std product extremely wide. Fused LayerNorm IBP
    // clamps normalized outputs to [-sqrt(n-1), sqrt(n-1)]; the decomposed
    // CROWN path must honor the same theoretical range.
    let lower = ArrayD::from_shape_vec(IxDyn(&[n]), vec![0.25_f32; n]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[n]), vec![0.35_f32; n]).unwrap();
    let x_ibp = BoundedTensor::new(lower, upper).unwrap();

    let identity = BatchedLinearBounds::identity(x_ibp.shape()).unwrap();
    let crown_result =
        decomposed_norm_crown_backward(&identity, &ny, &beta, eps, &x_ibp, false).unwrap();
    let concretized = crown_result.bounds.concretize_sound(&x_ibp).unwrap();

    let fused_ibp = LayerNormLayer::new(ny, beta, eps)
        .unwrap()
        .propagate_ibp(&x_ibp)
        .unwrap();

    for i in 0..n {
        assert!(
            concretized.lower()[[i]] >= fused_ibp.lower()[[i]] - 1e-4,
            "lower[{i}] escaped fused LayerNorm IBP envelope: crown={} ibp={}",
            concretized.lower()[[i]],
            fused_ibp.lower()[[i]],
        );
        assert!(
            concretized.upper()[[i]] <= fused_ibp.upper()[[i]] + 1e-4,
            "upper[{i}] escaped fused LayerNorm IBP envelope: crown={} ibp={}",
            concretized.upper()[[i]],
            fused_ibp.upper()[[i]],
        );
    }
}

/// Core soundness verification: given CROWN result from decomposed normalization
/// backward, verify that for sampled concrete inputs the true LayerNorm output
/// falls within the concretized bounds.
fn verify_decomposed_norm_soundness(
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    num_interior_samples: usize,
) -> Result<(), TestCaseError> {
    let shape = x_ibp.shape();
    let n = *shape.last().unwrap_or(&0);
    if n == 0 {
        return Err(TestCaseError::fail(
            "decomposed LayerNorm soundness oracle requires a nonempty normalized axis",
        ));
    }

    // Identity incoming A (output = input passthrough)
    let identity = BatchedLinearBounds::identity(shape)
        .map_err(|e| TestCaseError::fail(format!("identity creation failed: {e}")))?;

    // Run decomposed normalization CROWN backward
    let crown_result = decomposed_norm_crown_backward(&identity, ny, beta, eps, x_ibp, false)
        .map_err(|e| TestCaseError::fail(format!("decomposed_norm_crown_backward failed: {e}")))?;

    // Concretize: compute [lb, ub] by optimizing A@x+b over x in [x_l, x_u]
    let concretized = crown_result
        .bounds
        .concretize_sound(x_ibp)
        .map_err(|e| TestCaseError::fail(format!("concretize_sound failed: {e}")))?;

    let lb: Vec<f32> = concretized.lower().iter().copied().collect();
    let ub: Vec<f32> = concretized.upper().iter().copied().collect();

    // Sample concrete inputs and verify LayerNorm output is within bounds
    let x_lower: Vec<f32> = x_ibp.lower().iter().copied().collect();
    let x_upper: Vec<f32> = x_ibp.upper().iter().copied().collect();

    // Generate sample points per dimension
    let per_dim_samples: Vec<Vec<f32>> = (0..n)
        .map(|i| sample_points(x_lower[i], x_upper[i], num_interior_samples))
        .collect();

    // Test corners (all-lower, all-upper) and interior samples
    let mut test_points: Vec<Vec<f32>> = Vec::new();

    // All-lower corner
    test_points.push(x_lower.clone());
    // All-upper corner
    test_points.push(x_upper.clone());

    // Per-dimension extremes: each dimension at its lower/upper while others at midpoint
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

    // Interior grid: sample each dimension independently and combine
    // (full grid is exponential, so we sample along each axis instead)
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
        let true_output = layernorm(&x_arr, ny, beta, eps);

        for i in 0..n {
            let fx = true_output[i];
            // Scale tolerance by output magnitude to handle large values
            let scale_tol = DECOMPOSED_NORM_TOLERANCE * fx.abs().max(1.0);

            prop_assert!(
                lb[i] <= fx + scale_tol,
                "Decomposed norm CROWN lower bound violated at dim {i}: \
                 lb={} > f(x)={fx} (tol={scale_tol}, ny={}, eps={eps})",
                lb[i],
                ny[i],
            );
            prop_assert!(
                ub[i] + scale_tol >= fx,
                "Decomposed norm CROWN upper bound violated at dim {i}: \
                 ub={} < f(x)={fx} (tol={scale_tol}, ny={}, eps={eps})",
                ub[i],
                ny[i],
            );
        }
    }

    Ok(())
}

// =============================================================================
// PROPTEST: DECOMPOSED NORMALIZATION CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Decomposed LayerNorm CROWN soundness with tight perturbation (n=3).
    ///
    /// Verifies: for any concrete x in [center-hw, center+hw], the true
    /// LayerNorm output is within the CROWN-concretized bounds.
    ///
    /// Small n=3 tests the core McCormick + fan-out composition without
    /// excessive per-case compute cost.
    ///
    /// Part of #318.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_norm_crown_identity_tight_n3(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw in 0.01f32..0.2,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        verify_decomposed_norm_soundness(&ny, &beta, eps, &x_ibp, 20)?;
    }

    /// Decomposed norm CROWN with wider perturbation (n=3, hw up to 0.5).
    ///
    /// Tests McCormick relaxation quality with wider input intervals where
    /// the bilinear d*inv_std approximation is less tight.
    ///
    /// Part of #318.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_norm_crown_wide_perturbation_n3(
        c0 in -1.0f32..1.0,
        c1 in -1.0f32..1.0,
        c2 in -1.0f32..1.0,
        hw in 0.1f32..0.5,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        verify_decomposed_norm_soundness(&ny, &beta, eps, &x_ibp, 20)?;
    }

    /// Decomposed norm CROWN with larger dimension (n=8) and tight perturbation.
    ///
    /// Tests scaling: as n grows, the mean-subtraction backward Jacobian
    /// (δ_{ik} - 1/n) spreads A-matrix contributions across more elements.
    /// The variance path (Sqr→mean→Sqrt→Reciprocal) also depends on n.
    ///
    /// Part of #318.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_norm_crown_identity_tight_n8(
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
        b_scale in -0.5f32..0.5,
    ) {
        let n = 8;
        // Use uniform ny/beta scaled by the proptest parameter to reduce
        // dimension of the search space while still varying parameters.
        let ny = Array1::from_elem(n, g_scale);
        let beta = Array1::from_elem(n, b_scale);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        verify_decomposed_norm_soundness(&ny, &beta, eps, &x_ibp, 30)?;
    }

    /// Decomposed norm CROWN with asymmetric incoming A matrices.
    ///
    /// Tests non-identity incoming bounds to verify that the McCormick
    /// weight-sign-dependent plane selection works correctly when weights
    /// are negative. This is critical: McCormick lower/upper plane selection
    /// flips when the incoming coefficient is negative.
    ///
    /// Part of #318.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_norm_crown_negcoeff_n3(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw in 0.01f32..0.15,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
        w0 in -2.0f32..2.0,
        w1 in -2.0f32..2.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let x_ibp = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower_v).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper_v).unwrap(),
        ).unwrap();

        // Construct asymmetric incoming A: 2 output dims with mixed-sign weights
        // A_lower = [[w0, 0, w1], [0, w0, 0]]
        // A_upper = [[w0, 0, w1], [0, w0, 0]]
        // This tests McCormick plane selection when weight signs vary.
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

        // Reject degenerate all-zero A matrices — they have no negative
        // coefficients to test. Using prop_assume! so proptest generates
        // replacement cases instead of counting these as passed.
        prop_assume!(
            w0.abs() > f32::EPSILON || w1.abs() > f32::EPSILON,
            "Degenerate: both weights ~0, no negative coefficients to test"
        );

        let result = decomposed_norm_crown_backward(&incoming, &ny, &beta, eps, &x_ibp, false)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "decomposed_norm_crown_backward failed unexpectedly: {e}"
                ))
            })?;

        let concretized: BoundedTensor = result.bounds.concretize_sound(&x_ibp)
            .map_err(|e| TestCaseError::fail(
                format!("concretize failed: {e}")
            ))?;

        let lb: Vec<f32> = concretized.lower().iter().copied().collect();
        let ub: Vec<f32> = concretized.upper().iter().copied().collect();

        // Sample concrete inputs using sample_points (proper diverse coverage)
        let x_lower: Vec<f32> = x_ibp.lower().iter().copied().collect();
        let x_upper: Vec<f32> = x_ibp.upper().iter().copied().collect();
        let per_dim_samples: Vec<Vec<f32>> = (0..n)
            .map(|i| sample_points(x_lower[i], x_upper[i], 20))
            .collect();

        // Corners: all-lower, all-upper
        let mut test_points: Vec<Vec<f32>> = vec![x_lower.clone(), x_upper.clone()];

        // Per-dimension extremes at midpoint
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

        // Interior grid: vary each sample using loop index (not deterministic)
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
            let true_out = layernorm(&x_arr, &ny, &beta, eps);

            // Compute incoming transform of true output
            for o in 0..out_dim {
                let mut true_val = 0.0_f32;
                for j in 0..n {
                    true_val += incoming.lower_a()[[o, j]] * true_out[j];
                }
                true_val += incoming.lower_b()[o];

                let scale_tol = DECOMPOSED_NORM_TOLERANCE * true_val.abs().max(1.0);

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

    /// Decomposed norm CROWN with truly asymmetric incoming A (lower_a != upper_a).
    ///
    /// The negcoeff test above uses A_lower = A_upper (shared w0, w1). This test
    /// uses separate weights for lower and upper A matrices, exercising the
    /// McCormick plane selection code paths where lower and upper A entries
    /// differ. This is the case that arises in multi-layer CROWN: upstream
    /// layers produce different linear coefficients for lower vs upper bounds.
    ///
    /// Critical: lower bound must use lower_a @ LayerNorm(x) + lower_b,
    /// upper bound must use upper_a @ LayerNorm(x) + upper_b (independent).
    ///
    /// Part of #318. Directed by Prover P1-1074.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_decomposed_norm_crown_asymmetric_a_n3(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw in 0.01f32..0.15,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
        w_l0 in -2.0f32..2.0,
        w_l1 in -2.0f32..2.0,
        w_u0 in -2.0f32..2.0,
        w_u1 in -2.0f32..2.0,
    ) {
        let n = 3;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
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
        // contradictory (e.g., lower_a=2, upper_a=-1 with positive LayerNorm output).
        let center_arr = Array1::from_vec(centers.to_vec());
        let center_out = layernorm(&center_arr, &ny, &beta, eps);
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

        let result = decomposed_norm_crown_backward(&incoming, &ny, &beta, eps, &x_ibp, false)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "decomposed_norm_crown_backward failed unexpectedly: {e}"
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
            let true_out = layernorm(&x_arr, &ny, &beta, eps);

            for o in 0..out_dim {
                // Lower bound: lb <= lower_a @ LayerNorm(x) + lower_b
                let mut true_val_lower = 0.0_f32;
                for j in 0..n {
                    true_val_lower += incoming.lower_a()[[o, j]] * true_out[j];
                }
                true_val_lower += incoming.lower_b()[o];

                // Upper bound: ub >= upper_a @ LayerNorm(x) + upper_b
                let mut true_val_upper = 0.0_f32;
                for j in 0..n {
                    true_val_upper += incoming.upper_a()[[o, j]] * true_out[j];
                }
                true_val_upper += incoming.upper_b()[o];

                let scale_tol_l = DECOMPOSED_NORM_TOLERANCE * true_val_lower.abs().max(1.0);
                let scale_tol_u = DECOMPOSED_NORM_TOLERANCE * true_val_upper.abs().max(1.0);

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
