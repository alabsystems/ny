// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::SkipMergeLayer;

fn build_nan_firewall_network_2576() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::SkipMerge(SkipMergeLayer::new()));
    network
}

fn nan_input_bounds_2576() -> BoundedTensor {
    BoundedTensor::new_unchecked(
        arr1(&[1.0_f32, f32::NAN, -1.0]).into_dyn(),
        arr1(&[2.0_f32, f32::NAN, 0.5]).into_dyn(),
    )
    .expect("shape-only NaN fixture")
}

fn assert_nan_numerical_instability_2576<T: std::fmt::Debug>(
    result: Result<T>,
    expected_context: &str,
) {
    let err = result.expect_err("NaN bounds must fail fast");
    match err {
        NyError::NumericalInstability(msg) => {
            assert!(msg.contains("NaN"), "error should mention NaN, got: {msg}");
            assert!(
                msg.contains(expected_context),
                "error should mention '{expected_context}', got: {msg}"
            );
        }
        other => panic!("expected NumericalInstability, got: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_network_propagate_ibp_nan_input_returns_error_2576() {
    let network = build_nan_firewall_network_2576();
    let nan_input = nan_input_bounds_2576();

    assert_nan_numerical_instability_2576(network.propagate_ibp(&nan_input), "Sequential IBP");
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_network_propagate_ibp_sound_nan_input_returns_error_2576() {
    let network = build_nan_firewall_network_2576();
    let nan_input = nan_input_bounds_2576();

    assert_nan_numerical_instability_2576(
        network.propagate_ibp_sound(&nan_input),
        "Sequential IBP (sound)",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_network_collect_ibp_bounds_nan_input_returns_error_2576() {
    let network = build_nan_firewall_network_2576();
    let nan_input = nan_input_bounds_2576();

    assert_nan_numerical_instability_2576(
        network.collect_ibp_bounds(&nan_input),
        "Sequential IBP collect",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_network_collect_ibp_bounds_sound_nan_input_returns_error_2576() {
    let network = build_nan_firewall_network_2576();
    let nan_input = nan_input_bounds_2576();

    assert_nan_numerical_instability_2576(
        network.collect_ibp_bounds_sound(&nan_input),
        "Sequential IBP collect (sound)",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_network_collect_crown_ibp_bounds_nan_input_returns_error_2576() {
    let network = build_nan_firewall_network_2576();
    let nan_input = nan_input_bounds_2576();

    assert_nan_numerical_instability_2576(
        network.collect_crown_ibp_bounds(&nan_input),
        "Sequential IBP collect",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_network_collect_crown_ibp_precomputed_nan_returns_error_2576() {
    let network = build_nan_firewall_network_2576();
    let finite_input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let precomputed_nan =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("shape-only NaN fixture");

    assert_nan_numerical_instability_2576(
        network.collect_crown_ibp_bounds_with_precomputed_ibp(
            &finite_input,
            vec![precomputed_nan],
            None,
            None,
        ),
        "Sequential CROWN-IBP",
    );
}
