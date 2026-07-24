// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Verify the short-seq identity-RoPE Qwen3-TTS talker attention softmax
/// output through `Verifier::verify_graph(...)`. Softmax outputs are always in
/// [0, 1].
///
/// Uses IBP on the short sequence lane (seq_len=4) for runtime stability.
///
/// Part of #3654.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_softmax_range_verifier_smoke_3654() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (softmax_graph, softmax_name, input, output_size) =
        talker_verifier::talker_softmax_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Ibp,
        "talker softmax IBP verifier smoke",
    );

    talker_verifier::assert_softmax_verifier_result(
        &result,
        output_size,
        &softmax_name,
        "talker softmax IBP verifier smoke",
    );
    assert_verified_result_contains_center(
        &result,
        &softmax_graph,
        &input,
        "talker softmax IBP verifier smoke",
    );
}

/// Verify the short-seq talker softmax root smoke does not certify an
/// impossible property. Softmax outputs are non-negative, so requiring every
/// output to stay at or below `-0.1` must return `Unknown(BoundsTooLoose)`.
///
/// Part of #4061.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_softmax_impossible_spec_verifier_smoke_4061() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let label = "talker softmax impossible-spec IBP verifier smoke";
    let (softmax_graph, _softmax_name, input, output_size) =
        talker_verifier::talker_softmax_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size)
        .map(|_| Bound::new_allow_infinite(f32::NEG_INFINITY, -0.1))
        .collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Ibp,
        label,
    );

    match &result {
        VerificationResult::Unknown { bounds, reason, .. } => {
            assert!(
                matches!(reason, UnknownReason::BoundsTooLoose { .. }),
                "{label}: expected BoundsTooLoose, got {reason:?}"
            );
            common::assert_unknown_verifier_bounds_sound(bounds, output_size, label);
            assert!(
                bounds.iter().any(|bound| bound.upper() > -0.1),
                "{label}: expected at least one upper bound above impossible limit"
            );
        }
        _ => panic!("{label}: expected Unknown, got {result:?}"),
    }

    let concrete = common::evaluate_graph_at_center(&softmax_graph, &input, label);
    let concrete_flat = concrete.flatten();
    assert!(
        concrete_flat.lower().iter().any(|&value| value > -0.1),
        "{label}: expected center output to violate impossible upper bound"
    );
}
