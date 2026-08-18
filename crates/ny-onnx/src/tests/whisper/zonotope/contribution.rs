// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::*;
use super::super::helpers::{whisper_tiny_encoder, whisper_zero_input};

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_layernorm_forward_mode_request_fails_closed_without_bounds() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 4, 0.001);
    let config = MultiBlockConfig::default().with_layernorm_forward_mode(true);

    match whisper.verify_block_compositional_gpu_with_config(0, &input, None, &config) {
        Err(ny_core::NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("heuristic LayerNorm forward mode"));
            assert!(message.contains("no bounds were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("heuristic LayerNorm request returned untagged bounds"),
    }

    match whisper.attention_layernorm_output_ibp(0, &input, true) {
        Err(ny_core::NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("heuristic Whisper LayerNorm forward mode"));
            assert!(message.contains("no bounds were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("heuristic attention LayerNorm request returned untagged bounds"),
    }
}
