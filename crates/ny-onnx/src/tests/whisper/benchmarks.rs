#![cfg(feature = "benchmarks")]

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::fixtures::*;
use super::super::*;

// Run with: cargo test -p ny-onnx --features benchmarks whisper::benchmarks -- --nocapture

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_forward_mode_benchmark() {
    // Benchmark comparing conservative LayerNorm vs forward-mode (default) on Whisper.
    //
    // Forward mode (default) computes mean/std from the center point of input bounds,
    // which dramatically reduces bound explosion but is only approximately sound.
    //
    // This test measures:
    // 1. Bound width reduction (default vs conservative)
    // 2. Number of blocks verifiable before overflow
    // 3. Performance comparison
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);

    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;
    let num_blocks = whisper.encoder_layers;

    println!("\n=== LayerNorm Forward Mode Benchmark ===");
    println!("Model: Whisper-tiny");
    println!("Hidden dim: {}", hidden_dim);
    println!("Encoder blocks: {}", num_blocks);

    // Test multiple epsilon values
    let epsilons = [0.001, 0.01, 0.05, 0.1];
    let batch = 1;
    let seq_len = 4;

    println!("\n| Epsilon | Conservative | Forward Mode | Reduction | Con. Blocks | Fwd Blocks | Notes |");
    println!(
        "|---------|--------------|--------------|-----------|-------------|------------|-------|"
    );

    for epsilon in epsilons {
        let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
        let input = BoundedTensor::from_epsilon(input_data, epsilon).expect("valid test input");

        // Conservative mode (strictly sound but explodes)
        let conservative_config = MultiBlockConfig::conservative().with_terminate_on_overflow(true);
        let conservative_result = whisper.verify_encoder_sequential_with_config(
            &input,
            0,
            num_blocks,
            false,
            false,
            None,
            &conservative_config,
        );

        // Forward mode (default - practical verification)
        let forward_config = MultiBlockConfig::default().with_terminate_on_overflow(true);
        let forward_result = whisper.verify_encoder_sequential_with_config(
            &input,
            0,
            num_blocks,
            false,
            false,
            None,
            &forward_config,
        );

        let (conservative_width, conservative_blocks, con_overflow) = match &conservative_result {
            Ok((_, details)) => {
                let w = details.final_output_width;
                (w, details.blocks_completed, details.overflow_at_block)
            }
            Err(err) => {
                panic!(
                    "conservative verification failed at ε={}: {:?}",
                    epsilon, err
                );
            }
        };

        let (forward_width, forward_blocks, fwd_overflow) = match &forward_result {
            Ok((_, details)) => {
                let w = details.final_output_width;
                (w, details.blocks_completed, details.overflow_at_block)
            }
            Err(err) => {
                panic!("forward verification failed at ε={}: {:?}", epsilon, err);
            }
        };

        // Format width strings, showing "overflow" for NaN/Inf or when overflow detected
        let con_str = if conservative_width.is_nan()
            || !conservative_width.is_finite()
            || con_overflow.is_some()
        {
            format!("overflow@{}", con_overflow.unwrap_or(0))
        } else {
            format!("{:.2e}", conservative_width)
        };

        let fwd_str =
            if forward_width.is_nan() || !forward_width.is_finite() || fwd_overflow.is_some() {
                format!("overflow@{}", fwd_overflow.unwrap_or(0))
            } else {
                format!("{:.2e}", forward_width)
            };

        let reduction = if forward_width > 0.0 && forward_width.is_finite() {
            if conservative_width.is_finite() && conservative_width > 0.0 {
                format!("{:.1}x", conservative_width / forward_width)
            } else {
                "inf".to_string()
            }
        } else {
            "-".to_string()
        };

        // Check forward mode improvement
        let fwd_better = match (con_overflow.is_some(), fwd_overflow.is_some()) {
            (true, false) => "fwd wins",
            (false, true) => "con wins",
            (false, false) if forward_width < conservative_width => "fwd tighter",
            (false, false) => "similar",
            (true, true) => "both fail",
        };

        println!(
            "| {:.3} | {:>12} | {:>12} | {:>9} | {:>11} | {:>10} | {} |",
            epsilon, con_str, fwd_str, reduction, conservative_blocks, forward_blocks, fwd_better
        );
    }

    // Detailed single-epsilon test for assertions
    let epsilon = 0.01;
    let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
    let input = BoundedTensor::from_epsilon(input_data, epsilon).expect("valid test input");

    // Forward mode (default) should allow more blocks without overflow
    let forward_config = MultiBlockConfig::default().with_max_width(1e30); // High threshold to see how far we get
    let forward_result = whisper.verify_encoder_sequential_with_config(
        &input,
        0,
        num_blocks,
        false,
        false,
        None,
        &forward_config,
    );

    match forward_result {
        Ok((_, details)) => {
            println!("\n--- Forward Mode Details (ε={}) ---", epsilon);
            println!(
                "Blocks completed: {}/{}",
                details.blocks_completed, num_blocks
            );
            println!("Final bound width: {:.2e}", details.final_output_width);
            println!("Time: {}ms", details.total_time_ms);

            // With forward mode, we expect tighter bounds
            // Forward mode should complete more blocks or have smaller final width
            assert!(
                details.final_output_width < 1e30 || details.blocks_completed >= 2,
                "Forward mode should produce usable bounds"
            );
        }
        Err(e) => {
            panic!("Forward mode verification failed: {:?}", e);
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sequence_length_scaling() {
    // Benchmark how verification scales with sequence length.
    //
    // The complexity of attention is O(seq^2) for Q@K^T matmul,
    // so verification time should scale similarly.
    //
    // This test measures:
    // 1. Time scaling with sequence length
    // 2. Memory scaling (bound tensor sizes)
    // 3. Bound width scaling
    use ndarray::ArrayD;
    use ny_tensor::BoundedTensor;

    let path = require_test_model_with_hint("whisper_tiny_encoder.onnx", WHISPER_TEST_MODEL_HINT);

    let whisper = load_whisper(&path).expect("Failed to load model");
    let hidden_dim = whisper.hidden_dim;

    println!("\n=== Sequence Length Scaling Benchmark ===");
    println!("Model: Whisper-tiny (hidden_dim={})", hidden_dim);
    println!("Epsilon: 0.01");
    println!("Blocks: 4");
    println!("Mode: Forward-mode LayerNorm");

    // Test sequence lengths: 4, 8, 16, 32, 64
    let seq_lengths = [4, 8, 16, 32, 64];
    let batch = 1;
    let epsilon = 0.01f32;
    let num_blocks = 4;

    println!("\n| Seq Len | Time (ms) | Bound Width | Width/Prev | Time/Prev | Bound Size (KB) |");
    println!("|---------|-----------|-------------|------------|-----------|-----------------|");

    let mut prev_time = 0u128;
    let mut prev_width = 0.0f32;

    for seq_len in seq_lengths {
        let input_data = ArrayD::from_elem(ndarray::IxDyn(&[batch, seq_len, hidden_dim]), 0.0f32);
        let input = BoundedTensor::from_epsilon(input_data, epsilon).expect("valid test input");

        let config = MultiBlockConfig::default().with_terminate_on_overflow(true);

        let start = std::time::Instant::now();
        let result = whisper.verify_encoder_sequential_with_config(
            &input, 0, num_blocks, false, false, None, &config,
        );
        let elapsed = start.elapsed().as_millis();

        match result {
            Ok((output, details)) => {
                let width = details.final_output_width;

                // Calculate bound tensor memory (lower + upper, f32)
                let bound_size_kb = (seq_len * hidden_dim * 2 * 4) / 1024;

                // Scaling factors
                let time_ratio = if prev_time > 0 {
                    elapsed as f64 / prev_time as f64
                } else {
                    1.0
                };
                let width_ratio = if prev_width > 0.0 {
                    width / prev_width
                } else {
                    1.0
                };

                println!(
                    "| {:>7} | {:>9} | {:>11.2e} | {:>10.2}x | {:>9.2}x | {:>15} |",
                    seq_len, elapsed, width, width_ratio, time_ratio, bound_size_kb
                );

                prev_time = elapsed;
                prev_width = width;

                // Sanity check: output should have correct shape
                let out_shape = output.shape();
                assert_eq!(out_shape[0], batch, "Batch dim mismatch");
                assert_eq!(out_shape[1], seq_len, "Seq dim mismatch");
                assert_eq!(out_shape[2], hidden_dim, "Hidden dim mismatch");
            }
            Err(e) => {
                println!(
                    "| {:>7} | {:>9} | {:>11} | {:>10} | {:>9} | {:>15} |",
                    seq_len, "-", "ERROR", "-", "-", "-"
                );
                panic!("Verification failed at seq_len={}: {:?}", seq_len, e);
            }
        }
    }

    // Calculate approximate scaling exponent from first and last measurements
    // If T ~ seq^k, then k = log(T_last/T_first) / log(seq_last/seq_first)
    println!("\nNote: Time scaling between seq=4 and seq=64 gives approximate complexity.");
}
