// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::speaker_encoder::verifier_smoke as speaker_verifier;
use super::shared::{
    assert_verified_result_contains_center, run_verifier_smoke_route, VerifierSmokeRoute,
};

/// Verify the real ECAPA-TDNN speaker encoder norm-squared head through
/// `Verifier::verify_graph(...)`. norm_sq is always non-negative.
///
/// Uses IBP for runtime stability on the full real-model graph.
///
/// Part of #3654.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_speaker_norm_sq_verifier_smoke_3654() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let (norm_sq_graph, input) = speaker_verifier::speaker_norm_sq_verifier_setup();

    let output_bounds = vec![ny_core::Bound::new_allow_infinite(0.0, f32::INFINITY)];
    let result = run_verifier_smoke_route(
        &norm_sq_graph,
        &input,
        output_bounds,
        300_000,
        VerifierSmokeRoute::Ibp,
        "speaker norm_sq verifier smoke",
    );

    speaker_verifier::assert_scalar_nonnegative_verified(&result, "speaker norm_sq verifier smoke");

    assert_verified_result_contains_center(
        &result,
        &norm_sq_graph,
        &input,
        "speaker norm_sq verifier smoke",
    );
}

/// Verify the speaker encoder norm-squared head via GraphModel round-trip
/// through `Verifier::verify_graph(...)`.
///
/// This is the speaker analogue of the Kokoro round-trip verifier smoke
/// (`#4100`). It exercises the `OnnxModel -> GraphModel -> GraphNetwork`
/// path that builder-style downstream consumers use, then verifies the
/// same scalar `norm_sq >= 0` property through the shared route harness.
///
/// Part of #4179.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_speaker_norm_sq_graph_model_round_trip_verifier_smoke_4179() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let (norm_sq_graph, input) =
        speaker_verifier::speaker_graph_model_round_trip_norm_sq_verifier_setup();

    let output_bounds = vec![ny_core::Bound::new_allow_infinite(0.0, f32::INFINITY)];
    let result = run_verifier_smoke_route(
        &norm_sq_graph,
        &input,
        output_bounds,
        300_000,
        VerifierSmokeRoute::Ibp,
        "speaker GraphModel round-trip verifier smoke",
    );

    speaker_verifier::assert_scalar_nonnegative_verified(
        &result,
        "speaker norm_sq GraphModel round-trip verifier smoke",
    );

    assert_verified_result_contains_center(
        &result,
        &norm_sq_graph,
        &input,
        "speaker GraphModel round-trip verifier smoke",
    );
}
