// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::tests_support::{
    eval_variance_affine, interpolate, mean_square_bounds, positive_interval,
    true_variance_chain_value,
};
use super::variance_chain::accumulate_variance_chain;
use crate::layers::activations::LinearRelaxation;
use crate::layers::arithmetic::sqrt_linear_relaxation;
use crate::layers::misc::reciprocal::reciprocal_linear_relaxation;
use proptest::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_variance_chain_positive_coefficient_soundness() {
    let primary_lower = [-2.0, 0.5, 1.0];
    let primary_upper = [-1.0, 1.25, 2.0];
    let (mean_sq_lower, mean_sq_upper) = mean_square_bounds(&primary_lower, &primary_upper);
    let recip_relax = reciprocal_linear_relaxation(mean_sq_lower.sqrt(), mean_sq_upper.sqrt());
    let sqrt_relax = sqrt_linear_relaxation(mean_sq_lower, mean_sq_upper);
    let lower_aux_coeff = 0.75;
    let upper_aux_coeff = 0.95;
    let mut lower_primary = vec![0.0; primary_lower.len()];
    let mut upper_primary = vec![0.0; primary_lower.len()];
    let mut lower_bias = 0.0;
    let mut upper_bias = 0.0;

    let flags = accumulate_variance_chain(
        lower_aux_coeff,
        upper_aux_coeff,
        &recip_relax,
        &sqrt_relax,
        &primary_lower,
        &primary_upper,
        primary_lower.len(),
        0.0,
        &mut lower_primary,
        &mut upper_primary,
        &mut lower_bias,
        &mut upper_bias,
    );

    assert_eq!(flags, (false, false));
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let sample = [
            interpolate(primary_lower[0], primary_upper[0], t),
            interpolate(primary_lower[1], primary_upper[1], 1.0 - t),
            interpolate(primary_lower[2], primary_upper[2], 0.5),
        ];
        let lower = eval_variance_affine(&lower_primary, lower_bias, &sample);
        let upper = eval_variance_affine(&upper_primary, upper_bias, &sample);
        let true_lower = true_variance_chain_value(lower_aux_coeff, &sample);
        let true_upper = true_variance_chain_value(upper_aux_coeff, &sample);
        assert!(
            lower <= true_lower + 1e-6,
            "t={t}: lower bound {lower} > true {true_lower} + tol"
        );
        assert!(
            upper >= true_upper - 1e-6,
            "t={t}: upper bound {upper} < true {true_upper} - tol"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_variance_chain_nonfinite_propagation() {
    let recip_relax = LinearRelaxation::new(f32::INFINITY, 0.0, f32::INFINITY, 0.0);
    let sqrt_relax = LinearRelaxation::identity();
    let primary_lower = [0.5, 1.0];
    let primary_upper = [1.0, 1.5];
    let mut lower_primary = vec![0.0; primary_lower.len()];
    let mut upper_primary = vec![0.0; primary_lower.len()];
    let mut lower_bias = 0.0;
    let mut upper_bias = 0.0;

    let flags = accumulate_variance_chain(
        1.0,
        1.0,
        &recip_relax,
        &sqrt_relax,
        &primary_lower,
        &primary_upper,
        primary_lower.len(),
        0.0,
        &mut lower_primary,
        &mut upper_primary,
        &mut lower_bias,
        &mut upper_bias,
    );

    assert_eq!(flags, (true, true));
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    #[test]
    fn proptest_variance_chain_contains_true_output(
        (l0, u0) in positive_interval(),
        (l1, u1) in positive_interval(),
        (l2, u2) in positive_interval(),
        aux_coeff in 0.1f64..2.0,
        t0 in 0.0f32..1.0,
        t1 in 0.0f32..1.0,
        t2 in 0.0f32..1.0,
    ) {
        let primary_lower = [l0, l1, l2];
        let primary_upper = [u0, u1, u2];
        let (mean_sq_lower, mean_sq_upper) = mean_square_bounds(&primary_lower, &primary_upper);
        let recip_relax = reciprocal_linear_relaxation(mean_sq_lower.sqrt(), mean_sq_upper.sqrt());
        let sqrt_relax = sqrt_linear_relaxation(mean_sq_lower, mean_sq_upper);
        let mut lower_primary = vec![0.0; 3];
        let mut upper_primary = vec![0.0; 3];
        let mut lower_bias = 0.0;
        let mut upper_bias = 0.0;

        let flags = accumulate_variance_chain(
            aux_coeff,
            aux_coeff,
            &recip_relax,
            &sqrt_relax,
            &primary_lower,
            &primary_upper,
            3,
            0.0,
            &mut lower_primary,
            &mut upper_primary,
            &mut lower_bias,
            &mut upper_bias,
        );
        prop_assert_eq!(flags, (false, false));

        let sample = [
            interpolate(l0, u0, t0),
            interpolate(l1, u1, t1),
            interpolate(l2, u2, t2),
        ];
        let lower = eval_variance_affine(&lower_primary, lower_bias, &sample);
        let upper = eval_variance_affine(&upper_primary, upper_bias, &sample);
        let true_value = true_variance_chain_value(aux_coeff, &sample);
        prop_assert!(lower <= true_value + 1e-5,
            "lower bound {} > true {} + tol", lower, true_value);
        prop_assert!(upper >= true_value - 1e-5,
            "upper bound {} < true {} - tol", upper, true_value);
    }
}
