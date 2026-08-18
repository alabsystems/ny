// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::fixtures::*;
use super::super::*;
#[cfg(feature = "benchmarks")]
use super::compositional_fixture_3450::minimal_whisper_gpu_compositional_fixture_3450;
use super::helpers::debug_graph_ibp_failure;
#[cfg(feature = "benchmarks")]
use ny_gpu::{Backend, ComputeDevice};

fn assert_bounds_bitwise_equal(
    lhs: &ny_tensor::BoundedTensor,
    rhs: &ny_tensor::BoundedTensor,
    label: &str,
) {
    assert_eq!(lhs.shape(), rhs.shape(), "{label}: shape mismatch");
    assert!(
        lhs.lower()
            .iter()
            .zip(rhs.lower().iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: lower bounds differ"
    );
    assert!(
        lhs.upper()
            .iter()
            .zip(rhs.upper().iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: upper bounds differ"
    );
}

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_compositional_named_api_matches_conservative_graph_ibp() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    // The retained compositional API currently delegates to one conservative
    // full-block graph-IBP pass. Pin that fallback contract exactly.
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);

    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;

    // Create input tensor
    let batch = 1;
    let seq_len = std::cmp::min(
        4,
        WhisperModel::GPU_ATTENTION_THRESHOLD
            .saturating_sub(1)
            .max(1),
    );
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    println!("\n=== Compositional-name fallback vs conservative graph IBP ===");
    println!("Input shape: {:?}, epsilon: 0.01", input.shape());

    let mut full_graph = whisper
        .encoder_layer_graph_full(0)
        .expect("Failed to extract full graph");
    full_graph.set_layernorm_forward_mode(false);
    let graph_output = full_graph.propagate_ibp(&input).unwrap_or_else(|e| {
        let debug = debug_graph_ibp_failure(&full_graph, &input);
        panic!("Conservative graph IBP should succeed: {:?}\n{}", e, debug);
    });

    let (compat_output, details) = whisper
        .verify_block_compositional(0, &input)
        .unwrap_or_else(|e| panic!("Compatibility verification failed: {:?}", e));

    assert_bounds_bitwise_equal(&compat_output, &graph_output, "compositional-name fallback");
    let final_width = compat_output.max_width();
    for (stage, width) in [
        ("attention_delta_width", details.attention_delta_width),
        ("x_attn_width", details.x_attn_width),
        ("mlp_delta_width", details.mlp_delta_width),
        ("output_width", details.output_width),
    ] {
        assert_eq!(
            width.to_bits(),
            final_width.to_bits(),
            "{stage} must alias the final graph-IBP width in the fallback"
        );
    }
    assert!(
        !details.stage_metrics_available,
        "aliased fallback widths must not be advertised as stage measurements"
    );
}

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_direct_block_apis_reject_malformed_input_without_bounds() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    use ndarray::ArrayD;
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let rank_two = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[4, whisper.hidden_dim]), 0.0f32),
        0.01,
    )
    .expect("valid bounded tensor");
    let wrong_hidden = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, 4, whisper.hidden_dim + 1]), 0.0f32),
        0.01,
    )
    .expect("valid bounded tensor");

    for (label, result) in [
        (
            "rank-two default API",
            whisper.verify_block_compositional(0, &rank_two).map(|_| ()),
        ),
        (
            "rank-two explicit-config API",
            whisper
                .verify_block_compositional_gpu_with_config(
                    0,
                    &rank_two,
                    None,
                    &MultiBlockConfig::conservative(),
                )
                .map(|_| ()),
        ),
    ] {
        match result {
            Err(NyError::InvalidSpec(message)) => {
                assert!(
                    message.contains("[batch, sequence, hidden]"),
                    "{label}: {message}"
                );
            }
            Err(other) => panic!("{label}: expected InvalidSpec, got {other:?}"),
            Ok(()) => panic!("{label}: malformed input returned bounds"),
        }
    }

    match whisper.verify_block_compositional(0, &wrong_hidden) {
        Err(NyError::ShapeMismatch { expected, got }) => {
            assert_eq!(expected, vec![1, 4, whisper.hidden_dim]);
            assert_eq!(got, vec![1, 4, whisper.hidden_dim + 1]);
        }
        Err(other) => panic!("wrong-hidden input: expected ShapeMismatch, got {other:?}"),
        Ok(_) => panic!("wrong-hidden input returned bounds"),
    }
}

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_crown_named_compatibility_api_fails_closed_without_result() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    use ndarray::ArrayD;
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let input = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, 4, whisper.hidden_dim]), 0.0f32),
        0.01,
    )
    .expect("valid test input");

    match whisper.verify_block_compositional_crown(0, &input) {
        Err(NyError::UnsupportedConfiguration(message)) => {
            assert!(
                message.contains("no bounds were produced"),
                "unexpected error message: {message}"
            );
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => {
            panic!("the unavailable CROWN API must not return graph-IBP bounds as if CROWN ran")
        }
    }
}

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_unsupported_config_requests_fail_closed_without_bounds() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    use ndarray::ArrayD;
    use ny_propagate::layers::LayerNormCrownMode;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let input = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, 4, whisper.hidden_dim]), 0.0f32),
        0.01,
    );
    let input = input.expect("valid test input");

    let mut unsupported_requests = MultiBlockConfig::default()
        .with_max_width(1.0)
        .with_terminate_on_overflow(true)
        .with_layernorm_forward_mode(true)
        .with_layernorm_crown_mode(LayerNormCrownMode::IbpValidated)
        .with_zonotope_attention(true)
        .with_reset_zonotope_between_blocks(true)
        .with_crown_block_wise(true);
    unsupported_requests.continue_after_overflow = true;
    unsupported_requests.overflow_clamp_value = 17.0;

    match whisper.verify_block_compositional_gpu_with_config(0, &input, None, &unsupported_requests)
    {
        Err(ny_core::NyError::UnsupportedConfiguration(message)) => {
            for request in [
                "max_bound_width",
                "terminate_on_overflow",
                "continue_after_overflow",
                "overflow_clamp_value",
                "heuristic LayerNorm forward mode",
                "LayerNorm CROWN",
                "zonotope attention",
                "zonotope block reset",
                "block-wise CROWN",
                "no bounds were produced",
            ] {
                assert!(
                    message.contains(request),
                    "error did not identify {request}: {message}"
                );
            }
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("unsupported execution requests must not return fallback bounds"),
    }
}

#[cfg(feature = "benchmarks")]
#[ntest::timeout(60000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_attention_ibp_projection_lookup_3450() {
    // Regression #3450: building the real Whisper-tiny attention subgraph must
    // locate its Q/K/V/out projection MatMuls. The historical
    // `attention_ibp_gpu` entry point no longer exists; the builder lookup is
    // exercised before this helper narrows the graph output to LayerNorm.
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;

    let input = BoundedTensor::from_epsilon(
        ArrayD::from_elem(
            ndarray::IxDyn(&[1, WhisperModel::GPU_ATTENTION_THRESHOLD, hidden_dim]),
            0.0f32,
        ),
        0.01,
    )
    .expect("valid test input");

    let ln_output = whisper
        .attention_layernorm_output_ibp(0, &input, false)
        .unwrap_or_else(|e| panic!("attention subgraph construction should succeed: {e:?}"));

    assert!(
        ln_output
            .lower()
            .iter()
            .zip(ln_output.upper().iter())
            .all(|(l, u)| l <= u),
        "attention LayerNorm bounds must be ordered"
    );
}

#[cfg(feature = "benchmarks")]
#[ntest::timeout(60000)]
#[test]
fn test_gpu_aware_api_rejects_unavailable_device_request() {
    // A supplied device is an execution request. The current Whisper backend
    // must fail closed instead of silently substituting CPU graph IBP.
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let whisper = minimal_whisper_gpu_compositional_fixture_3450();
    let hidden_dim = whisper.hidden_dim;

    // Create input tensor at the GPU attention threshold.
    let batch = 1;
    let seq_len = WhisperModel::GPU_ATTENTION_THRESHOLD;
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, 0.01).expect("valid test input");

    let cpu_device =
        ComputeDevice::new(Backend::Cpu).expect("the always-available CPU device must initialize");
    let (cpu_output, cpu_details) = whisper
        .verify_block_compositional_gpu(0, &input, None)
        .unwrap_or_else(|e| panic!("CPU graph-IBP compatibility call failed: {e:?}"));
    let (conservative_output, _) = whisper
        .verify_block_compositional(0, &input)
        .unwrap_or_else(|e| panic!("conservative graph-IBP call failed: {e:?}"));
    assert_bounds_bitwise_equal(&cpu_output, &conservative_output, "no-config GPU-aware API");
    assert!(!cpu_details.used_gpu_attention);
    assert!(!cpu_details.used_zonotope_attention);
    assert!(!cpu_details.stage_metrics_available);

    match whisper.verify_block_compositional_gpu(0, &input, Some(&cpu_device)) {
        Err(ny_core::NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("GPU execution"));
            assert!(message.contains("no bounds were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("an unavailable GPU request must not return CPU fallback bounds"),
    }
}

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_sequential_verification_apis_fail_closed_without_result() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    use ndarray::ArrayD;
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);
    let whisper = load_whisper(&path).expect("Failed to load model");
    let input = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, 2, whisper.hidden_dim]), 0.0f32),
        0.01,
    )
    .expect("valid test input");

    let calls = [
        (
            "explicit-config API",
            whisper.verify_encoder_sequential_with_config(
                &input,
                0,
                whisper.encoder_layers,
                false,
                false,
                None,
                &MultiBlockConfig::default(),
            ),
        ),
        (
            "default-config wrapper",
            whisper.verify_encoder_sequential(
                &input,
                0,
                whisper.encoder_layers,
                false,
                false,
                None,
            ),
        ),
        (
            "full-encoder wrapper",
            whisper.verify_full_encoder(&input, false, false, None),
        ),
    ];

    for (label, result) in calls {
        match result {
            Err(NyError::UnsupportedConfiguration(message)) => {
                assert!(
                    message.contains("no bounds or completion metadata were produced"),
                    "{label}: unexpected error message: {message}"
                );
            }
            Err(other) => panic!("{label}: expected UnsupportedConfiguration, got {other:?}"),
            Ok(_) => panic!("{label}: unavailable sequential verification returned a result"),
        }
    }
}
