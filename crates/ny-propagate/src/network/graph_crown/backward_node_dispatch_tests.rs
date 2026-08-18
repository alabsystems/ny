// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::backward_node_dispatch::{backward_div_to_numerator, DivBackwardResult};
use crate::{BoundedTensor, LinearBounds};
use ndarray::{arr1, arr2};

fn div_overflow_fixture_3438() -> (LinearBounds, BoundedTensor, BoundedTensor, BoundedTensor) {
    let node_lb = LinearBounds::new(
        arr2(&[[f32::MAX]]),
        arr1(&[0.0_f32]),
        arr2(&[[f32::MAX]]),
        arr1(&[0.0_f32]),
    )
    .expect("node linear bounds should be well-formed");
    let input_a_bounds =
        BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("numerator bounds should be valid");
    let input_b_bounds = BoundedTensor::new(
        arr1(&[1.0e-38_f32]).into_dyn(),
        arr1(&[2.0e-38_f32]).into_dyn(),
    )
    .expect("denominator bounds should be valid");
    let node_output_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("output bounds should be valid");
    (node_lb, input_a_bounds, input_b_bounds, node_output_bounds)
}

fn assert_conservative_scalar_bounds_3438(bounds: &LinearBounds, label: &str) {
    assert_eq!(
        bounds.lower_a()[[0, 0]],
        0.0,
        "{label} lower A should fall back to conservative zero coefficients"
    );
    assert_eq!(
        bounds.upper_a()[[0, 0]],
        0.0,
        "{label} upper A should fall back to conservative zero coefficients"
    );
    assert!(
        bounds.lower_b()[0].is_infinite() && bounds.lower_b()[0].is_sign_negative(),
        "{label} lower bias should fall back to -Inf, got {}",
        bounds.lower_b()[0]
    );
    assert!(
        bounds.upper_b()[0].is_infinite() && bounds.upper_b()[0].is_sign_positive(),
        "{label} upper bias should fall back to +Inf, got {}",
        bounds.upper_b()[0]
    );
}

/// Negative (sign-definite) denominator: reciprocal scaling must stay sound.
///
/// Dense-sample the input box and assert the propagated affine lower/upper
/// form (in terms of the numerator x, with the denominator y free in its box)
/// encloses x/y for the identity spec and for a negated spec.
#[test]
fn test_graph_crown_div_negative_denominator_sound() {
    // y strictly negative: [-4, -1]. x in [-2, 3].
    let x_lo = -2.0_f32;
    let x_hi = 3.0_f32;
    let y_lo = -4.0_f32;
    let y_hi = -1.0_f32;

    let input_a = BoundedTensor::new(arr1(&[x_lo]).into_dyn(), arr1(&[x_hi]).into_dyn()).unwrap();
    let input_b = BoundedTensor::new(arr1(&[y_lo]).into_dyn(), arr1(&[y_hi]).into_dyn()).unwrap();
    // Div output range (sound enclosure not needed precisely for the test).
    let out =
        BoundedTensor::new(arr1(&[-3.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();

    for &sign in &[1.0_f32, -1.0_f32] {
        let node_lb = LinearBounds::new(
            arr2(&[[sign]]),
            arr1(&[0.0_f32]),
            arr2(&[[sign]]),
            arr1(&[0.0_f32]),
        )
        .unwrap();

        let result =
            backward_div_to_numerator(&node_lb, &input_a, &input_b, &out).expect("Div backward");
        let DivBackwardResult::PropagateNumerator(bounds) = result else {
            panic!("sign-definite denominator should propagate numerator bounds (sign={sign})");
        };

        let la = bounds.lower_a()[[0, 0]];
        let ua = bounds.upper_a()[[0, 0]];
        let lb = bounds.lower_b()[0];
        let ub = bounds.upper_b()[0];

        let steps = 60;
        for i in 0..=steps {
            let x = x_lo + (x_hi - x_lo) * (i as f32 / steps as f32);
            for j in 0..=steps {
                let y = y_lo + (y_hi - y_lo) * (j as f32 / steps as f32);
                // The propagated bound is expressed in terms of the numerator x;
                // the denominator dependence is absorbed into the bias envelope.
                let z = sign * (x / y);
                let lo = la * x + lb;
                let hi = ua * x + ub;
                let tol = 1e-3 * (1.0 + z.abs());
                assert!(
                    lo <= z + tol,
                    "neg-denom lower {lo} > {z} at (x={x}, y={y}, sign={sign})"
                );
                assert!(
                    hi >= z - tol,
                    "neg-denom upper {hi} < {z} at (x={x}, y={y}, sign={sign})"
                );
            }
        }
    }
}

#[test]
fn test_graph_crown_div_non_power_reciprocal_with_negative_numerator_sound() {
    let x_lo = -2.0_f32.powi(30);
    let x_hi = -2.0_f32.powi(20);
    let y_lo = 0.1_f32;
    let y_hi = 0.3_f32;
    let input_a = BoundedTensor::new(arr1(&[x_lo]).into_dyn(), arr1(&[x_hi]).into_dyn()).unwrap();
    let input_b = BoundedTensor::new(arr1(&[y_lo]).into_dyn(), arr1(&[y_hi]).into_dyn()).unwrap();
    let out =
        BoundedTensor::new(arr1(&[-2.0e10_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn()).unwrap();
    let node_lb = LinearBounds::new(
        arr2(&[[1.000_000_1_f32]]),
        arr1(&[0.0]),
        arr2(&[[1.000_000_1_f32]]),
        arr1(&[0.0]),
    )
    .unwrap();
    let DivBackwardResult::PropagateNumerator(bounds) =
        backward_div_to_numerator(&node_lb, &input_a, &input_b, &out).unwrap()
    else {
        panic!("strictly positive denominator should propagate")
    };

    let (la, lb) = (
        f64::from(bounds.lower_a()[[0, 0]]),
        f64::from(bounds.lower_b()[0]),
    );
    let (ua, ub) = (
        f64::from(bounds.upper_a()[[0, 0]]),
        f64::from(bounds.upper_b()[0]),
    );
    for ix in 0..=64 {
        let x = f64::from(x_lo) + (f64::from(x_hi) - f64::from(x_lo)) * (ix as f64 / 64.0);
        for iy in 0..=64 {
            let y = f64::from(y_lo) + (f64::from(y_hi) - f64::from(y_lo)) * (iy as f64 / 64.0);
            let truth = f64::from(1.000_000_1_f32) * x / y;
            assert!(la * x + lb <= truth, "lower missed at x={x:e}, y={y:e}");
            assert!(ua * x + ub >= truth, "upper missed at x={x:e}, y={y:e}");
        }
    }
}

#[test]
fn test_graph_crown_div_discharges_incoming_coefficient_error() {
    let input_a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    let input_b = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let output = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    let mut node_lb =
        LinearBounds::new(arr2(&[[1.0]]), arr1(&[0.0]), arr2(&[[1.0]]), arr1(&[0.0])).unwrap();
    node_lb.set_coeff_err(arr2(&[[0.25]]), arr2(&[[0.25]]));

    let DivBackwardResult::PropagateNumerator(bounds) =
        backward_div_to_numerator(&node_lb, &input_a, &input_b, &output).unwrap()
    else {
        panic!("point positive denominator should propagate")
    };
    assert!(
        bounds.lower_b()[0] <= -0.5,
        "lower bias={}",
        bounds.lower_b()[0]
    );
    assert!(
        bounds.upper_b()[0] >= 0.5,
        "upper bias={}",
        bounds.upper_b()[0]
    );
    assert!(!bounds.has_coeff_err());
}

#[test]
fn test_graph_crown_div_nonfinite_denominator_falls_back() {
    let input_a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let input_b = BoundedTensor::new_allow_infinite(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();
    let output = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let node_lb = LinearBounds::identity(1);
    assert!(matches!(
        backward_div_to_numerator(&node_lb, &input_a, &input_b, &output).unwrap(),
        DivBackwardResult::ConcretizeCurrentNode(_)
    ));
}

/// Mixed-sign denominator (0 ∈ [ly, uy]) must keep the concretization fallback.
#[test]
fn test_graph_crown_div_mixed_sign_denominator_concretizes() {
    let input_a =
        BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
    let input_b =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let out =
        BoundedTensor::new(arr1(&[-5.0_f32]).into_dyn(), arr1(&[5.0_f32]).into_dyn()).unwrap();
    let node_lb = LinearBounds::new(
        arr2(&[[1.0_f32]]),
        arr1(&[0.0_f32]),
        arr2(&[[1.0_f32]]),
        arr1(&[0.0_f32]),
    )
    .unwrap();
    let result =
        backward_div_to_numerator(&node_lb, &input_a, &input_b, &out).expect("Div backward");
    assert!(
        matches!(result, DivBackwardResult::ConcretizeCurrentNode(_)),
        "mixed-sign denominator must concretize, not propagate"
    );
}

#[test]
fn test_graph_crown_div_sign_definite_elements_may_have_different_signs() {
    let input_a = BoundedTensor::new(
        arr1(&[-2.0_f32, -3.0]).into_dyn(),
        arr1(&[4.0_f32, 5.0]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        arr1(&[1.0_f32, -4.0]).into_dyn(),
        arr1(&[2.0_f32, -1.0]).into_dyn(),
    )
    .unwrap();
    let out = BoundedTensor::new(
        arr1(&[-5.0_f32, -5.0]).into_dyn(),
        arr1(&[5.0_f32, 5.0]).into_dyn(),
    )
    .unwrap();
    let node_lb = LinearBounds::identity(2);
    let DivBackwardResult::PropagateNumerator(bounds) =
        backward_div_to_numerator(&node_lb, &input_a, &input_b, &out).unwrap()
    else {
        panic!("element-wise sign-definite denominator should propagate")
    };

    for element in 0..2 {
        let (x_lo, x_hi) = (input_a.lower()[[element]], input_a.upper()[[element]]);
        let (y_lo, y_hi) = (input_b.lower()[[element]], input_b.upper()[[element]]);
        for ix in 0..=32 {
            let x = x_lo + (x_hi - x_lo) * ix as f32 / 32.0;
            for iy in 0..=32 {
                let y = y_lo + (y_hi - y_lo) * iy as f32 / 32.0;
                let truth = x / y;
                let lower = bounds.lower_a().row(element).dot(&arr1(&[
                    if element == 0 { x } else { 0.0 },
                    if element == 1 { x } else { 0.0 },
                ])) + bounds.lower_b()[element];
                let upper = bounds.upper_a().row(element).dot(&arr1(&[
                    if element == 0 { x } else { 0.0 },
                    if element == 1 { x } else { 0.0 },
                ])) + bounds.upper_b()[element];
                assert!(
                    lower <= truth,
                    "element={element} lower={lower} truth={truth}"
                );
                assert!(
                    upper >= truth,
                    "element={element} upper={upper} truth={truth}"
                );
            }
        }
    }
}

#[test]
fn test_graph_crown_div_rejects_incompatible_denominator_broadcast() {
    let input_a = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[2.0, 3.0]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
        arr1(&[2.0_f32, 3.0, 4.0]).into_dyn(),
    )
    .unwrap();
    let out = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[3.0, 3.0]).into_dyn(),
    )
    .unwrap();
    assert!(matches!(
        backward_div_to_numerator(&LinearBounds::identity(2), &input_a, &input_b, &out).unwrap(),
        DivBackwardResult::ConcretizeCurrentNode(_)
    ));
}

#[test]
fn test_graph_crown_div_overflow_falls_back_3438() {
    let (node_lb, input_a_bounds, input_b_bounds, node_output_bounds) = div_overflow_fixture_3438();
    let result = backward_div_to_numerator(
        &node_lb,
        &input_a_bounds,
        &input_b_bounds,
        &node_output_bounds,
    )
    .expect("Div backward should succeed");
    let DivBackwardResult::PropagateNumerator(bounds) = result else {
        panic!("positive denominators with matching shapes should propagate numerator bounds");
    };

    assert_conservative_scalar_bounds_3438(&bounds, "graph-CROWN Div");
}
