// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Verify the canonical real-RoPE Qwen3-TTS talker attention softmax output
/// through the bounded IBP verifier route. Softmax outputs are always in
/// [0, 1].
///
/// Short-sequence and boundary tests cover the CROWN route without the
/// uninterruptible seven-minute full-spec pass that previously lived here.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_talker_softmax_real_rope_verifier_smoke_l1() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let (softmax_graph, softmax_name, input, output_size) =
        talker_verifier::talker_softmax_real_rope_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Ibp,
        "talker softmax real-RoPE IBP verifier smoke",
    );

    talker_verifier::assert_softmax_verifier_result(
        &result,
        output_size,
        &softmax_name,
        "talker softmax real-RoPE IBP verifier smoke",
    );
    assert_verified_result_contains_center(
        &result,
        &softmax_graph,
        &input,
        "talker softmax real-RoPE IBP verifier smoke",
    );
}
