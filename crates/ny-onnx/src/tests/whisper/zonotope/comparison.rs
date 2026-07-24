// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::*;
use super::super::helpers::{whisper_tiny_encoder, whisper_zero_input};
use crate::GpuCompositionalDetails;

fn print_ibp_vs_zonotope_block(ibp: &GpuCompositionalDetails, zono: &GpuCompositionalDetails) {
    println!("\nAttention delta bounds:");
    println!("  IBP:      {:.3e}", ibp.attention_delta_width);
    println!("  Zonotope: {:.3e}", zono.attention_delta_width);
    if zono.attention_delta_width > 0.0 && ibp.attention_delta_width > 0.0 {
        println!(
            "  Ratio (IBP/Zonotope): {:.1}x",
            ibp.attention_delta_width / zono.attention_delta_width
        );
    }
    println!("\nBlock output bounds:");
    println!("  IBP:      {:.3e}", ibp.output_width);
    println!("  Zonotope: {:.3e}", zono.output_width);
    if zono.output_width > 0.0 && ibp.output_width > 0.0 {
        println!(
            "  Ratio (IBP/Zonotope): {:.1}x",
            ibp.output_width / zono.output_width
        );
    }
    println!("\nDetails:");
    println!("  IBP used GPU attention: {}", ibp.used_gpu_attention);
    println!("  Zonotope enabled:       {}", zono.used_zonotope_attention);
}

fn assert_zonotope_within_ibp_envelope(
    ibp: &GpuCompositionalDetails,
    zono: &GpuCompositionalDetails,
) {
    assert!(
        zono.used_zonotope_attention,
        "Zonotope config should enable zonotope attention"
    );
    assert!(
        !ibp.used_zonotope_attention,
        "IBP config should not enable zonotope attention"
    );
    assert!(
        ibp.output_width.is_finite(),
        "IBP output bounds should be finite"
    );
    assert!(
        zono.output_width.is_finite(),
        "Zonotope output bounds should be finite"
    );
    assert!(
        zono.attention_delta_width <= ibp.attention_delta_width * 1.01,
        "Zonotope attention delta should stay within IBP after the Softmax cut: \
         zonotope={:.6e}, ibp={:.6e}",
        zono.attention_delta_width,
        ibp.attention_delta_width
    );
    assert!(
        zono.output_width <= ibp.output_width * 1.01,
        "Zonotope block output should stay within IBP after the Softmax cut: \
         zonotope={:.6e}, ibp={:.6e}",
        zono.output_width,
        ibp.output_width
    );
}

// Budget: shares the 33MB dynamo whisper fixture; the first test to touch the
// cached model/network pays the full debug-build load/convert cost inside its
// timer, which exceeds the old 10s budget under parallel suite load. 120s
// matches the heavy whisper siblings and still guards against hangs.
#[ntest::timeout(120000)]
#[test]
fn test_zonotope_attention_vs_ibp_encoder_block() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Compare the runtime zonotope attention lane against the IBP baseline on a
    // single Whisper encoder block. After #318 Packet B/C, the zonotope branch
    // intentionally cuts Softmax back to IBP, so the overall block bounds
    // should stay within the IBP envelope while preserving the Q/K tightening.
    let whisper = whisper_tiny_encoder();
    let input = whisper_zero_input(whisper.hidden_dim, 16, 0.001);

    println!("\n=== Zonotope vs IBP Attention Comparison ===");
    println!(
        "Input: batch=1, seq=16, hidden={}, eps=0.001",
        whisper.hidden_dim
    );

    let ibp_config = MultiBlockConfig::default().with_terminate_on_overflow(true);
    let ibp_result = whisper
        .verify_block_compositional_gpu_with_config(0, &input, None, &ibp_config)
        .unwrap_or_else(|e| panic!("IBP verification failed: {:?}", e));

    let zonotope_config = MultiBlockConfig::tightest_attention().with_terminate_on_overflow(true);
    let zonotope_result = whisper
        .verify_block_compositional_gpu_with_config(0, &input, None, &zonotope_config)
        .unwrap_or_else(|e| panic!("Zonotope verification failed: {:?}", e));

    print_ibp_vs_zonotope_block(&ibp_result.1, &zonotope_result.1);
    assert_zonotope_within_ibp_envelope(&ibp_result.1, &zonotope_result.1);
}

fn print_multiblock_row(block: usize, ibp_width: f32, zono_width: f32) {
    let ratio = if zono_width > 0.0 && zono_width.is_finite() {
        format!("{:.1}x", ibp_width / zono_width)
    } else {
        "-".to_string()
    };
    println!(
        "| {:5} | {:9.2e} | {:14.2e} | {:11} |",
        block, ibp_width, zono_width, ratio
    );
}

// Budget: shares the 33MB dynamo whisper fixture; the first test to touch the
// cached model/network pays the full debug-build load/convert cost inside its
// timer, which exceeds the old 10s budget under parallel suite load. 120s
// matches the heavy whisper siblings and still guards against hangs.
#[ntest::timeout(120000)]
#[test]
#[allow(unused_assignments)]
fn test_zonotope_attention_multiblock_improvement() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Test zonotope attention across multiple blocks to show compounding improvement
    let whisper = whisper_tiny_encoder();
    let num_blocks = whisper.encoder_layers.min(4);
    let input = whisper_zero_input(whisper.hidden_dim, 16, 0.001);

    println!("\n=== Zonotope vs IBP Multi-Block Comparison ===");
    println!(
        "Model: Whisper-tiny, {} blocks, seq=16, eps=0.001",
        num_blocks
    );
    println!("\n| Block | IBP Width | Zonotope Width | Improvement |");
    println!("|-------|-----------|----------------|-------------|");

    let ibp_config = MultiBlockConfig::default().with_terminate_on_overflow(false);
    let zonotope_config = MultiBlockConfig::tightest_attention().with_terminate_on_overflow(false);
    let mut ibp_current = input.clone();
    let mut zono_current = input;

    for block in 0..num_blocks {
        let ibp_result = whisper
            .verify_block_compositional_gpu_with_config(block, &ibp_current, None, &ibp_config)
            .unwrap_or_else(|e| panic!("IBP verification failed for block {}: {:?}", block, e));
        let zono_result = whisper
            .verify_block_compositional_gpu_with_config(
                block,
                &zono_current,
                None,
                &zonotope_config,
            )
            .unwrap_or_else(|e| {
                panic!("Zonotope verification failed for block {}: {:?}", block, e)
            });

        let ibp_width = ibp_result.0.max_width();
        let zono_width = zono_result.0.max_width();
        assert!(
            ibp_width.is_finite(),
            "IBP bounds not finite at block {block}"
        );
        assert!(
            zono_width.is_finite(),
            "Zonotope bounds not finite at block {block}"
        );
        print_multiblock_row(block, ibp_width, zono_width);

        ibp_current = ibp_result.0.clone();
        zono_current = zono_result.0.clone();
    }
}
