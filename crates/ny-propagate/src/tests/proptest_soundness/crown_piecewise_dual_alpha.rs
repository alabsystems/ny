// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness for dual-alpha ReLU CROWN backward (#3393).
//!
//! Tests `propagate_linear_with_alpha` with `Some(alpha_upper)`, exercising the
//! dual alpha path where lower and upper bound objectives use independent alpha
//! parameters. This is the core alpha-CROWN optimization loop's relaxation.
//!
//! For dual alpha to be meaningfully exercised:
//! - Crossing neurons (l < 0 < u) make alpha relevant
//! - la > 0 exercises alpha_lower (lower bound, positive incoming coeff)
//! - ua < 0 exercises alpha_upper (upper bound, negative incoming coeff)
//!
//! Reference: auto_LiRPA/operators/relu.py selected_alpha[0] (lower) vs [1] (upper)

use crate::layers::ReLULayer;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use proptest::prelude::*;

use super::{sample_points, CROWN_TOLERANCE};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // 1-neuron dual-alpha: crossing neuron with asymmetric incoming coefficients
    // =========================================================================

    /// 1 crossing neuron, asymmetric incoming bounds (c_l != c_u).
    /// Exercises: alpha_lower when c_l > 0, alpha_upper when c_u < 0.
    /// Soundness: linearized bounds must contain true ReLU output at all sample points.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_relu_dual_alpha_1neuron(
        l in -10.0f32..-0.01,
        d in 0.02f32..10.0,
        c_l in -5.0f32..5.0,
        c_u in -5.0f32..5.0,
        alpha_l in 0.0f32..1.0,
        alpha_u in 0.0f32..1.0,
    ) {
        let u = (l + d).min(10.0);
        prop_assume!(u > 0.01); // Must be crossing

        // At least one direction exercises alpha
        prop_assume!(c_l.abs() > 0.01 || c_u.abs() > 0.01);

        let layer = ReLULayer::new();
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![c_l]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![c_u]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let pre_activation = ny_tensor::BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let alpha = arr1(&[alpha_l]);
        let alpha_upper = arr1(&[alpha_u]);

        let (result, _grad_lower, _grad_upper) = layer
            .propagate_linear_with_alpha(&incoming, &pre_activation, &alpha, Some(&alpha_upper))
            .unwrap();

        for x in sample_points(l, u, 100) {
            let relu_x = x.max(0.0);

            // True objectives at this point
            let true_lower = (c_l as f64) * (relu_x as f64);
            let true_upper = (c_u as f64) * (relu_x as f64);

            // CROWN linearized bounds
            let lb = (result.lower_a[[0, 0]] as f64) * (x as f64) + (result.lower_b[0] as f64);
            let ub = (result.upper_a[[0, 0]] as f64) * (x as f64) + (result.upper_b[0] as f64);

            let max_intermediate = (result.lower_a[[0, 0]].abs() * x.abs())
                .max(result.upper_a[[0, 0]].abs() * x.abs())
                .max(1.0);
            let tol = (CROWN_TOLERANCE as f64) * (max_intermediate as f64);

            prop_assert!(
                lb <= true_lower + tol,
                "Dual-alpha lower violated at x={x}: lb={lb} > true={true_lower} (tol={tol}, \
                 c_l={c_l}, alpha_l={alpha_l}, alpha_u={alpha_u})"
            );
            prop_assert!(
                ub + tol >= true_upper,
                "Dual-alpha upper violated at x={x}: ub={ub} < true={true_upper} (tol={tol}, \
                 c_u={c_u}, alpha_l={alpha_l}, alpha_u={alpha_u})"
            );
        }
    }

    // =========================================================================
    // 2-neuron dual-alpha: mixed crossing/stable with asymmetric coefficients
    // =========================================================================

    /// 2 neurons (both crossing), asymmetric incoming bounds.
    /// Exercises dual alpha interaction across multiple neurons.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_relu_dual_alpha_2neuron(
        l0 in -10.0f32..-0.01,
        u0 in 0.02f32..10.0,
        l1 in -10.0f32..-0.01,
        u1 in 0.02f32..10.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        upper_c0 in -5.0f32..5.0,
        upper_c1 in -5.0f32..5.0,
        a0_l in 0.0f32..1.0,
        a1_l in 0.0f32..1.0,
    ) {
        // Use different upper alphas derived from lower to keep parameter count manageable
        let a0_u = 1.0 - a0_l;
        let a1_u = 1.0 - a1_l;

        // l < 0 < u guaranteed by ranges above — both neurons are crossing

        // At least one coefficient exercises alpha
        prop_assume!(
            lower_c0.abs() > 0.01 || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01 || upper_c1.abs() > 0.01
        );

        let layer = ReLULayer::new();
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let pre_activation = ny_tensor::BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let alpha = arr1(&[a0_l, a1_l]);
        let alpha_upper = arr1(&[a0_u, a1_u]);

        let (result, _grad_lower, _grad_upper) = layer
            .propagate_linear_with_alpha(&incoming, &pre_activation, &alpha, Some(&alpha_upper))
            .unwrap();

        let samples_0 = sample_points(l0, u0, 20);
        let samples_1 = sample_points(l1, u1, 20);

        for &x0 in &samples_0 {
            for &x1 in &samples_1 {
                let r0 = x0.max(0.0) as f64;
                let r1 = x1.max(0.0) as f64;

                // True objectives
                let true_lower = (lower_c0 as f64) * r0 + (lower_c1 as f64) * r1;
                let true_upper = (upper_c0 as f64) * r0 + (upper_c1 as f64) * r1;

                // CROWN linearized bounds
                let lb = (result.lower_a[[0, 0]] as f64) * (x0 as f64)
                    + (result.lower_a[[0, 1]] as f64) * (x1 as f64)
                    + (result.lower_b[0] as f64);
                let ub = (result.upper_a[[0, 0]] as f64) * (x0 as f64)
                    + (result.upper_a[[0, 1]] as f64) * (x1 as f64)
                    + (result.upper_b[0] as f64);

                let max_intermediate = (result.lower_a[[0, 0]].abs() * x0.abs())
                    .max(result.lower_a[[0, 1]].abs() * x1.abs())
                    .max(result.upper_a[[0, 0]].abs() * x0.abs())
                    .max(result.upper_a[[0, 1]].abs() * x1.abs())
                    .max(1.0);
                let tol = (CROWN_TOLERANCE as f64) * (max_intermediate as f64);

                prop_assert!(
                    lb <= true_lower + tol,
                    "2-neuron dual-alpha lower violated at ({x0}, {x1}): lb={lb} > \
                     true={true_lower} (tol={tol})"
                );
                prop_assert!(
                    ub + tol >= true_upper,
                    "2-neuron dual-alpha upper violated at ({x0}, {x1}): ub={ub} < \
                     true={true_upper} (tol={tol})"
                );
            }
        }
    }

    // =========================================================================
    // Dual alpha vs single alpha: verify single is a special case of dual
    // =========================================================================

    /// When alpha_upper == alpha_lower, dual and single produce identical results.
    /// This is a consistency check, not a soundness check.
    #[ntest::timeout(10000)]
    #[test]
    fn consistency_dual_alpha_degenerates_to_single(
        l in -10.0f32..-0.01,
        d in 0.02f32..10.0,
        c_l in -5.0f32..5.0,
        c_u in -5.0f32..5.0,
        alpha_val in 0.0f32..1.0,
    ) {
        let u = (l + d).min(10.0);
        prop_assume!(u > 0.01);
        prop_assume!(c_l.abs() > 0.01 || c_u.abs() > 0.01);

        let layer = ReLULayer::new();
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![c_l]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![c_u]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let pre_activation = ny_tensor::BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let alpha = arr1(&[alpha_val]);

        let (result_single, grad_l_single, grad_u_single) = layer
            .propagate_linear_with_alpha(&incoming, &pre_activation, &alpha, None)
            .unwrap();
        let (result_dual, grad_l_dual, grad_u_dual) = layer
            .propagate_linear_with_alpha(&incoming, &pre_activation, &alpha, Some(&alpha))
            .unwrap();

        // Coefficients must be identical (both paths use the same alpha value)
        prop_assert!(
            (result_single.lower_a[[0, 0]] - result_dual.lower_a[[0, 0]]).abs() < 1e-7,
            "lower_a mismatch: single={}, dual={}",
            result_single.lower_a[[0, 0]], result_dual.lower_a[[0, 0]]
        );
        prop_assert!(
            (result_single.upper_a[[0, 0]] - result_dual.upper_a[[0, 0]]).abs() < 1e-7,
            "upper_a mismatch: single={}, dual={}",
            result_single.upper_a[[0, 0]], result_dual.upper_a[[0, 0]]
        );
        prop_assert!(
            (result_single.lower_b[0] - result_dual.lower_b[0]).abs() < 1e-7,
            "lower_b mismatch: single={}, dual={}",
            result_single.lower_b[0], result_dual.lower_b[0]
        );
        prop_assert!(
            (result_single.upper_b[0] - result_dual.upper_b[0]).abs() < 1e-7,
            "upper_b mismatch: single={}, dual={}",
            result_single.upper_b[0], result_dual.upper_b[0]
        );
        // Gradients must match
        prop_assert!(
            (grad_l_single[0] - grad_l_dual[0]).abs() < 1e-7,
            "grad_lower mismatch: single={}, dual={}",
            grad_l_single[0], grad_l_dual[0]
        );
        prop_assert!(
            (grad_u_single[0] - grad_u_dual[0]).abs() < 1e-7,
            "grad_upper mismatch: single={}, dual={}",
            grad_u_single[0], grad_u_dual[0]
        );
    }
}
