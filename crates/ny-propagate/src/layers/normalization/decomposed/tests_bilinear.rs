// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::bilinear::accumulate_mccormick_bilinear_term;
use super::tests_support::{
    eval_bilinear_affine, expected_lower_plane, expected_upper_plane, interpolate,
    interval_samples, ordered_interval,
};
use ny_tensor::{next_down_f32, next_up_f32};
use proptest::prelude::*;

fn accumulate_bilinear_slots(
    lower_weight: f32,
    upper_weight: f32,
    primary_lower: f32,
    primary_upper: f32,
    aux_lower: f32,
    aux_upper: f32,
) -> ((bool, bool), (f64, f64, f64, f64, f64, f64)) {
    let mut lower_primary = 0.0;
    let mut upper_primary = 0.0;
    let mut lower_aux = 0.0;
    let mut upper_aux = 0.0;
    let mut lower_bias = 0.0;
    let mut upper_bias = 0.0;

    let flags = accumulate_mccormick_bilinear_term(
        lower_weight,
        upper_weight,
        primary_lower,
        primary_upper,
        aux_lower,
        aux_upper,
        &mut lower_primary,
        &mut upper_primary,
        &mut lower_aux,
        &mut upper_aux,
        &mut lower_bias,
        &mut upper_bias,
    );

    (
        flags,
        (
            lower_primary,
            upper_primary,
            lower_aux,
            upper_aux,
            lower_bias,
            upper_bias,
        ),
    )
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_bilinear_positive_weight_soundness() {
    let weight = 0.7;
    let primary_lower = 0.25;
    let primary_upper = 1.5;
    let aux_lower = 0.5;
    let aux_upper = 1.75;
    let (flags, (lower_primary, upper_primary, lower_aux, upper_aux, lower_bias, upper_bias)) =
        accumulate_bilinear_slots(
            weight,
            weight,
            primary_lower,
            primary_upper,
            aux_lower,
            aux_upper,
        );

    assert_eq!(flags, (false, false));
    for primary in interval_samples(primary_lower, primary_upper) {
        for aux in interval_samples(aux_lower, aux_upper) {
            let true_value = f64::from(weight) * f64::from(primary) * f64::from(aux);
            let lower = eval_bilinear_affine(lower_primary, lower_aux, lower_bias, primary, aux);
            let upper = eval_bilinear_affine(upper_primary, upper_aux, upper_bias, primary, aux);
            assert!(
                lower <= true_value + 1e-9,
                "lower={lower} true={true_value} primary={primary} aux={aux}"
            );
            assert!(
                upper >= true_value - 1e-9,
                "upper={upper} true={true_value} primary={primary} aux={aux}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_bilinear_negative_weight_soundness() {
    let weight = -0.85;
    let primary_lower = 0.4;
    let primary_upper = 1.25;
    let aux_lower = 0.3;
    let aux_upper = 1.4;
    let (flags, (lower_primary, upper_primary, lower_aux, upper_aux, lower_bias, upper_bias)) =
        accumulate_bilinear_slots(
            weight,
            weight,
            primary_lower,
            primary_upper,
            aux_lower,
            aux_upper,
        );

    assert_eq!(flags, (false, false));
    for primary in interval_samples(primary_lower, primary_upper) {
        for aux in interval_samples(aux_lower, aux_upper) {
            let true_value = f64::from(weight) * f64::from(primary) * f64::from(aux);
            let lower = eval_bilinear_affine(lower_primary, lower_aux, lower_bias, primary, aux);
            let upper = eval_bilinear_affine(upper_primary, upper_aux, upper_bias, primary, aux);
            assert!(
                lower <= true_value + 1e-9,
                "lower={lower} true={true_value} primary={primary} aux={aux}"
            );
            assert!(
                upper >= true_value - 1e-9,
                "upper={upper} true={true_value} primary={primary} aux={aux}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_bilinear_zero_weight() {
    let mut lower_primary = 1.0;
    let mut upper_primary = -2.0;
    let mut lower_aux = 3.0;
    let mut upper_aux = -4.0;
    let mut lower_bias = 5.0;
    let mut upper_bias = -6.0;

    let flags = accumulate_mccormick_bilinear_term(
        0.0,
        0.0,
        -1.0,
        2.0,
        0.5,
        1.5,
        &mut lower_primary,
        &mut upper_primary,
        &mut lower_aux,
        &mut upper_aux,
        &mut lower_bias,
        &mut upper_bias,
    );

    assert_eq!(flags, (false, false));
    assert_eq!(lower_primary, 1.0);
    assert_eq!(upper_primary, -2.0);
    assert_eq!(lower_aux, 3.0);
    assert_eq!(upper_aux, -4.0);
    assert_eq!(lower_bias, 5.0);
    assert_eq!(upper_bias, -6.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_bilinear_nonfinite_input_flags() {
    let mut lower_primary = 1.0;
    let mut upper_primary = 2.0;
    let mut lower_aux = 3.0;
    let mut upper_aux = 4.0;
    let mut lower_bias = 5.0;
    let mut upper_bias = 6.0;

    let flags = accumulate_mccormick_bilinear_term(
        0.5,
        1.5,
        f32::NAN,
        2.0,
        0.25,
        1.0,
        &mut lower_primary,
        &mut upper_primary,
        &mut lower_aux,
        &mut upper_aux,
        &mut lower_bias,
        &mut upper_bias,
    );

    assert_eq!(flags, (true, true));
    assert_eq!(lower_primary, 1.0);
    assert_eq!(upper_primary, 2.0);
    assert_eq!(lower_aux, 3.0);
    assert_eq!(upper_aux, 4.0);
    assert_eq!(lower_bias, 5.0);
    assert_eq!(upper_bias, 6.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_bilinear_directed_rounding() {
    let lower_weight = 0.35;
    let upper_weight = 0.9;
    let primary_lower = -0.7;
    let primary_upper = 1.1;
    let aux_lower = 0.2;
    let aux_upper = 1.7;
    let (_, (lower_primary, upper_primary, lower_aux, upper_aux, lower_bias, upper_bias)) =
        accumulate_bilinear_slots(
            lower_weight,
            upper_weight,
            primary_lower,
            primary_upper,
            aux_lower,
            aux_upper,
        );

    let (expected_lower_primary_coeff, expected_lower_aux_coeff, expected_lower_const) =
        expected_lower_plane(
            lower_weight,
            primary_lower,
            primary_upper,
            aux_lower,
            aux_upper,
        );
    let (expected_upper_primary_coeff, expected_upper_aux_coeff, expected_upper_const) =
        expected_upper_plane(
            upper_weight,
            primary_lower,
            primary_upper,
            aux_lower,
            aux_upper,
        );

    assert_eq!(
        lower_primary,
        f64::from(next_down_f32(lower_weight * expected_lower_primary_coeff))
    );
    assert_eq!(
        upper_primary,
        f64::from(next_up_f32(upper_weight * expected_upper_primary_coeff))
    );
    assert_eq!(
        lower_aux,
        f64::from(lower_weight) * f64::from(expected_lower_aux_coeff)
    );
    assert_eq!(
        upper_aux,
        f64::from(upper_weight) * f64::from(expected_upper_aux_coeff)
    );
    assert_eq!(
        lower_bias,
        f64::from(lower_weight) * f64::from(expected_lower_const)
    );
    assert_eq!(
        upper_bias,
        f64::from(upper_weight) * f64::from(expected_upper_const)
    );
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]

    #[test]
    fn proptest_mccormick_bilinear_contains_true_product(
        weight in -3.0f32..3.0,
        (primary_lower, primary_upper) in ordered_interval(),
        (aux_lower, aux_upper) in ordered_interval(),
        primary_t in 0.0f32..1.0,
        aux_t in 0.0f32..1.0,
    ) {
        let (flags, (lower_primary, upper_primary, lower_aux, upper_aux, lower_bias, upper_bias)) =
            accumulate_bilinear_slots(
            weight,
            weight,
            primary_lower,
            primary_upper,
            aux_lower,
            aux_upper,
        );
        prop_assert_eq!(flags, (false, false));

        let primary = interpolate(primary_lower, primary_upper, primary_t);
        let aux = interpolate(aux_lower, aux_upper, aux_t);
        let true_value = f64::from(weight) * f64::from(primary) * f64::from(aux);
        let lower = eval_bilinear_affine(lower_primary, lower_aux, lower_bias, primary, aux);
        let upper = eval_bilinear_affine(upper_primary, upper_aux, upper_bias, primary, aux);
        prop_assert!(lower <= true_value + 1e-6,
            "lower bound {} > true {} + tol (primary={}, aux={})", lower, true_value, primary, aux);
        prop_assert!(upper >= true_value - 1e-6,
            "upper bound {} < true {} - tol (primary={}, aux={})", upper, true_value, primary, aux);
    }
}
