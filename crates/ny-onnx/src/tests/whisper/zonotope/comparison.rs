// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::*;
use super::super::helpers::{whisper_tiny_encoder, whisper_zero_input};
use ny_core::NyError;

#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_zonotope_request_fails_closed_without_bounds() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 16, 0.001);
    let config = MultiBlockConfig::default()
        .with_zonotope_attention(true)
        .with_reset_zonotope_between_blocks(true);

    match whisper.verify_block_compositional_gpu_with_config(0, &input, None, &config) {
        Err(NyError::UnsupportedConfiguration(message)) => {
            assert!(message.contains("zonotope attention"));
            assert!(message.contains("zonotope block reset"));
            assert!(message.contains("no bounds were produced"));
        }
        Err(other) => panic!("expected UnsupportedConfiguration, got {other:?}"),
        Ok(_) => panic!("an unavailable zonotope request must not return graph-IBP bounds"),
    }
}
