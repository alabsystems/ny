// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2};
use ny_propagate::layers::{
    ArctanLayer, BoundPropagation, CosLayer, SigmoidLayer, SinLayer, TanLayer, TanhLayer,
};
use ny_propagate::LinearBounds;
use ny_tensor::BoundedTensor;

const TOL: f32 = 1e-3;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_crown_backward_sound_signed<F>(
    label: &str,
    layer: &dyn BoundPropagation,
    intervals: &[(f32, f32)],
    coeff: f32,
    f: F,
) where
    F: Fn(f32) -> f32,
{
    let bounds = LinearBounds::new(
        arr2(&[[coeff]]),
        arr1(&[0.0]),
        arr2(&[[coeff]]),
        arr1(&[0.0]),
    )
    .unwrap();

    for &(l, u) in intervals {
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn())
            .expect("invariant: singleton interval should be valid");
        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre)
            .unwrap_or_else(|e| {
                panic!("{label}: propagate_linear_with_bounds failed for [{l}, {u}]: {e}")
            });

        for k in 0..=80 {
            let t = k as f32 / 80.0;
            let x = l + (u - l) * t;
            let y = coeff * f(x);

            let lower = result.lower_a()[[0, 0]] * x + result.lower_b()[0];
            let upper = result.upper_a()[[0, 0]] * x + result.upper_b()[0];

            assert!(
                lower <= y + TOL,
                "{label}: lower violated on [{l}, {u}] at x={x}: lower={lower}, y={y}, coeff={coeff}"
            );
            assert!(
                upper >= y - TOL,
                "{label}: upper violated on [{l}, {u}] at x={x}: upper={upper}, y={y}, coeff={coeff}"
            );
        }
    }
}

fn assert_crown_backward_sound_two_neuron_mixed<F>(
    label: &str,
    layer: &dyn BoundPropagation,
    intervals: [(f32, f32); 2],
    coeffs: [f32; 2],
    f: F,
) where
    F: Fn(f32) -> f32,
{
    let ((l0, u0), (l1, u1)) = (intervals[0], intervals[1]);
    let bounds = LinearBounds::new(
        arr2(&[[coeffs[0], coeffs[1]]]),
        arr1(&[0.0]),
        arr2(&[[coeffs[0], coeffs[1]]]),
        arr1(&[0.0]),
    )
    .unwrap();

    let pre = BoundedTensor::new(arr1(&[l0, l1]).into_dyn(), arr1(&[u0, u1]).into_dyn())
        .expect("invariant: two-neuron interval should be valid");
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre)
        .unwrap_or_else(|e| {
            panic!(
                "{label}: propagate_linear_with_bounds failed for [{l0}, {u0}] x [{l1}, {u1}]: {e}"
            )
        });

    for i in 0..=40 {
        let t0 = i as f32 / 40.0;
        let x0 = l0 + (u0 - l0) * t0;
        for j in 0..=40 {
            let t1 = j as f32 / 40.0;
            let x1 = l1 + (u1 - l1) * t1;

            let y = coeffs[0] * f(x0) + coeffs[1] * f(x1);
            let lower =
                result.lower_a()[[0, 0]] * x0 + result.lower_a()[[0, 1]] * x1 + result.lower_b()[0];
            let upper =
                result.upper_a()[[0, 0]] * x0 + result.upper_a()[[0, 1]] * x1 + result.upper_b()[0];

            assert!(
                lower <= y + TOL,
                "{label}: mixed-sign lower violated at x0={x0}, x1={x1}: lower={lower}, y={y}"
            );
            assert!(
                upper >= y - TOL,
                "{label}: mixed-sign upper violated at x0={x0}, x1={x1}: upper={upper}, y={y}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn trig_layers_exercise_negative_coefficient_sign_swap_path_2307() {
    let negative_coeff = -1.0;

    let sin = SinLayer::new();
    let sin_intervals = [(0.1, 1.0), (3.4, 5.0), (-1.0, 1.0)];
    assert_crown_backward_sound_signed("sin", &sin, &sin_intervals, negative_coeff, f32::sin);

    let cos = CosLayer::new();
    let cos_intervals = [(0.1, 1.0), (2.0, 3.0), (-0.5, 1.5)];
    assert_crown_backward_sound_signed("cos", &cos, &cos_intervals, negative_coeff, f32::cos);

    let tan = TanLayer::new();
    let tan_intervals = [(-1.2, -0.2), (-0.5, 0.3), (0.2, 1.0)];
    assert_crown_backward_sound_signed("tan", &tan, &tan_intervals, negative_coeff, f32::tan);
}

#[ntest::timeout(10000)]
#[test]
fn s_shaped_layers_exercise_negative_coefficient_sign_swap_path_2307() {
    let negative_coeff = -1.0;

    let tanh = TanhLayer::new();
    let tanh_intervals = [(-3.0, 3.0), (-4.0, -1.0), (-1.5, 0.5), (0.5, 3.0)];
    assert_crown_backward_sound_signed("tanh", &tanh, &tanh_intervals, negative_coeff, f32::tanh);

    let sigmoid_layer = SigmoidLayer::new();
    let sigmoid_intervals = [(-3.0, 3.0), (-6.0, -2.0), (-1.0, 1.0), (1.0, 6.0)];
    assert_crown_backward_sound_signed(
        "sigmoid",
        &sigmoid_layer,
        &sigmoid_intervals,
        negative_coeff,
        sigmoid,
    );

    let arctan = ArctanLayer::new();
    let arctan_intervals = [(-3.0, 3.0), (-6.0, -1.5), (-1.0, 1.0), (0.5, 4.0)];
    assert_crown_backward_sound_signed(
        "arctan",
        &arctan,
        &arctan_intervals,
        negative_coeff,
        f32::atan,
    );
}

#[ntest::timeout(10000)]
#[test]
fn periodic_sin_mixed_sign_two_neuron_soundness_2307() {
    let sin = SinLayer::new();
    assert_crown_backward_sound_two_neuron_mixed(
        "sin-mixed",
        &sin,
        [(0.2, 0.9), (3.6, 4.6)],
        [-1.0, 1.0],
        f32::sin,
    );
}

#[ntest::timeout(10000)]
#[test]
fn periodic_cos_mixed_sign_two_neuron_soundness_2307() {
    let cos = CosLayer::new();
    // Concave region + convex region with mixed-sign coefficients
    assert_crown_backward_sound_two_neuron_mixed(
        "cos-mixed",
        &cos,
        [(0.2, 0.9), (2.2, 2.8)],
        [-1.0, 1.0],
        f32::cos,
    );
}

#[ntest::timeout(10000)]
#[test]
fn periodic_tan_mixed_sign_two_neuron_soundness_2307() {
    let tan = TanLayer::new();
    // Two intervals within (-π/2, π/2) to avoid asymptotes
    assert_crown_backward_sound_two_neuron_mixed(
        "tan-mixed",
        &tan,
        [(-0.8, -0.2), (0.2, 0.8)],
        [-1.0, 1.0],
        f32::tan,
    );
}

#[ntest::timeout(10000)]
#[test]
fn s_shaped_tanh_mixed_sign_two_neuron_soundness_2307() {
    let tanh = TanhLayer::new();
    assert_crown_backward_sound_two_neuron_mixed(
        "tanh-mixed",
        &tanh,
        [(-2.5, -0.5), (0.3, 2.1)],
        [-1.0, 1.0],
        f32::tanh,
    );
}

#[ntest::timeout(10000)]
#[test]
fn s_shaped_sigmoid_mixed_sign_two_neuron_soundness_2307() {
    let sigmoid_layer = SigmoidLayer::new();
    assert_crown_backward_sound_two_neuron_mixed(
        "sigmoid-mixed",
        &sigmoid_layer,
        [(-4.0, -1.0), (1.0, 4.0)],
        [-1.0, 1.0],
        sigmoid,
    );
}

#[ntest::timeout(10000)]
#[test]
fn s_shaped_arctan_mixed_sign_two_neuron_soundness_2307() {
    let arctan = ArctanLayer::new();
    assert_crown_backward_sound_two_neuron_mixed(
        "arctan-mixed",
        &arctan,
        [(-3.0, -0.5), (0.5, 3.0)],
        [-1.0, 1.0],
        f32::atan,
    );
}

// Asymmetric magnitude coefficients: [-2.0, 0.5] exercises different scaling
// on the positive vs negative branch of compose_lower/compose_upper.
#[ntest::timeout(10000)]
#[test]
fn sin_asymmetric_magnitude_two_neuron_soundness_2307() {
    let sin = SinLayer::new();
    assert_crown_backward_sound_two_neuron_mixed(
        "sin-asymmetric",
        &sin,
        [(0.2, 0.9), (3.6, 4.6)],
        [-2.0, 0.5],
        f32::sin,
    );
}

#[ntest::timeout(10000)]
#[test]
fn sigmoid_asymmetric_magnitude_two_neuron_soundness_2307() {
    let sigmoid_layer = SigmoidLayer::new();
    assert_crown_backward_sound_two_neuron_mixed(
        "sigmoid-asymmetric",
        &sigmoid_layer,
        [(-2.0, 0.5), (0.5, 3.0)],
        [-2.0, 0.5],
        sigmoid,
    );
}
