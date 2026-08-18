// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use super::helpers::{whisper_tiny_encoder, whisper_zero_input};
use std::time::Instant;

// This is a scaling smoke test for the production compatibility lane, not a
// comparative benchmark: unavailable zonotope requests fail closed and are
// covered by `zonotope/comparison.rs`.
#[ntest::timeout(120000)]
#[cfg(feature = "external-whisper")]
#[test]
fn test_whisper_block_ibp_sequence_scaling() {
    crate::test_fixtures::assert_test_model_available!("whisper_tiny_encoder.onnx");
    let whisper = whisper_tiny_encoder();
    let hidden_dim = whisper.hidden_dim;
    let epsilon = 0.001;

    println!("\n=== Whisper block CPU graph-IBP sequence scaling ===");
    println!("Model: Whisper-tiny (single block 0), eps={}", epsilon);
    println!("\n| Seq Len | Time (ms) | Output Width |");
    println!("|---------|-----------|--------------|");

    let config = MultiBlockConfig::default();

    for &seq_len in &[8, 16, 32, 64] {
        let input = whisper_zero_input(hidden_dim, seq_len, epsilon);

        let start = Instant::now();
        let (output, details) = whisper
            .verify_block_compositional_gpu_with_config(0, &input, None, &config)
            .unwrap_or_else(|e| panic!("CPU graph IBP failed: {:?}", e));
        let elapsed = start.elapsed().as_millis();
        let width = output.max_width();

        println!("| {:7} | {:9} | {:12.2e} |", seq_len, elapsed, width);
        assert_eq!(output.shape(), &[1, seq_len, hidden_dim]);
        assert!(
            output
                .lower()
                .iter()
                .zip(output.upper().iter())
                .all(|(lower, upper)| lower <= upper),
            "bounds must remain ordered at sequence length {seq_len}"
        );
        assert!(
            width.is_finite() && width > 0.0,
            "output width must be positive and finite at sequence length {seq_len}"
        );
        assert!(!details.used_gpu_attention);
        assert!(!details.used_zonotope_attention);
    }
}
