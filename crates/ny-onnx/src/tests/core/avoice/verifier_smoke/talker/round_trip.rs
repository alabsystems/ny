// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Verify the talker attention softmax through the GraphModel round-trip
/// path (ONNX -> GraphModel -> GraphNetwork -> Verifier).
///
/// Part of #3923.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_talker_softmax_graph_model_round_trip_verifier_smoke_3923() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let (softmax_graph, softmax_name, input, output_size) =
        talker_verifier::talker_graph_model_round_trip_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Ibp,
        "talker GraphModel round-trip verifier smoke",
    );
    talker_verifier::assert_softmax_verifier_result(
        &result,
        output_size,
        &softmax_name,
        "talker GraphModel round-trip verifier smoke",
    );
    assert_verified_result_contains_center(
        &result,
        &softmax_graph,
        &input,
        "talker GraphModel round-trip verifier smoke",
    );
}
