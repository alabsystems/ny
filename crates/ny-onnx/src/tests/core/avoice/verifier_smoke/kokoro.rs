// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::kokoro_vocoder::verifier_smoke as kokoro_verifier;
use super::shared::{
    assert_verified_result_contains_center, run_verifier_smoke_route, VerifierSmokeRoute,
};

/// Verify the real Kokoro vocoder prefix energy head through
/// `Verifier::verify_graph(...)`. Energy (sum of squares) is always >= 0.
///
/// Uses IBP on the shallowest prefix subgraph (up to first ConvTranspose1d)
/// with a scalar PowConstant(2.0) -> ReduceSum energy head appended.
///
/// Part of #4029.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_kokoro_prefix_energy_verifier_smoke_4029() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    let (energy_graph, input) = kokoro_verifier::kokoro_prefix_energy_verifier_setup();

    let output_bounds = vec![ny_core::Bound::new_allow_infinite(0.0, f32::INFINITY)];
    let result = run_verifier_smoke_route(
        &energy_graph,
        &input,
        output_bounds,
        300_000,
        VerifierSmokeRoute::Ibp,
        "kokoro prefix energy verifier smoke",
    );

    kokoro_verifier::assert_kokoro_prefix_energy_verifier_result(
        &result,
        "kokoro prefix energy IBP verifier smoke",
    );

    assert_verified_result_contains_center(
        &result,
        &energy_graph,
        &input,
        "kokoro prefix energy verifier smoke",
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_kokoro_prefix_energy_graph_model_round_trip_verifier_smoke_4100() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    let (energy_graph, input) =
        kokoro_verifier::kokoro_graph_model_round_trip_prefix_energy_verifier_setup();

    let output_bounds = vec![ny_core::Bound::new_allow_infinite(0.0, f32::INFINITY)];
    let result = run_verifier_smoke_route(
        &energy_graph,
        &input,
        output_bounds,
        300_000,
        VerifierSmokeRoute::Ibp,
        "kokoro GraphModel round-trip verifier smoke",
    );

    kokoro_verifier::assert_kokoro_prefix_energy_verifier_result(
        &result,
        "kokoro prefix energy GraphModel round-trip verifier smoke",
    );

    assert_verified_result_contains_center(
        &result,
        &energy_graph,
        &input,
        "kokoro GraphModel round-trip verifier smoke",
    );
}
