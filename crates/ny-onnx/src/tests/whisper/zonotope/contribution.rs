// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::*;
use super::super::helpers::{whisper_tiny_encoder, whisper_zero_input};
use std::time::Instant;

// Budget: three whole-block configs x 4 blocks at seq=16 measure ~24s in a
// debug build now that block extraction works against the dynamo fixture (the
// old 10s budget dated from when the test died fast at extraction). 120s
// matches the heavy whisper siblings and leaves headroom under parallel load.
#[ntest::timeout(120000)]
#[test]
fn test_optimization_contribution() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Isolate the contribution of each optimization:
    // 1. IBP + backward LN (conservative - strictly sound but explodes)
    // 2. IBP + forward LN (default - practical verification)
    // 3. Zonotope + forward LN (marginal additional tightening)
    let whisper = whisper_tiny_encoder();
    let hidden_dim = whisper.hidden_dim;
    let num_blocks = whisper.encoder_layers;
    let epsilon = 0.001;
    let seq_len = 16;
    let input = whisper_zero_input(hidden_dim, seq_len, epsilon);

    println!("\n=== Optimization Contribution Analysis ===");
    println!(
        "Model: Whisper-tiny, {} blocks, seq={}, hidden={}, eps={}",
        num_blocks, seq_len, hidden_dim, epsilon
    );
    println!("\n| Config                  | Time (ms) | Final Width | vs Baseline |");
    println!("|-------------------------|-----------|-------------|-------------|");

    // Config 1: IBP + backward LN (conservative baseline - strictly sound but explodes)
    let cfg_ibp_bw = MultiBlockConfig::conservative();

    // Config 2: IBP + forward LN (default config)
    let cfg_ibp_fw = MultiBlockConfig::default();

    // Config 3: Zonotope + forward LN (tightest)
    let cfg_zono = MultiBlockConfig::tightest_attention();

    let configs: &[(&str, &MultiBlockConfig)] = &[
        ("IBP + backward LN", &cfg_ibp_bw),
        ("IBP + forward LN", &cfg_ibp_fw),
        ("Zonotope + forward LN", &cfg_zono),
    ];

    let mut baseline_width = 0.0f32;

    for (name, config) in configs {
        let mut current = input.clone();
        let start = Instant::now();
        for block in 0..num_blocks {
            let (out, _) = whisper
                .verify_block_compositional_gpu_with_config(block, &current, None, config)
                .unwrap_or_else(|e| {
                    panic!(
                        "Compositional verification failed for block {}: {:?}",
                        block, e
                    )
                });
            current = out;
        }
        let elapsed = start.elapsed().as_millis();
        let final_width = current.max_width();

        // Set baseline
        if baseline_width == 0.0 {
            baseline_width = final_width;
        }

        let improvement = if final_width > 0.0 && final_width.is_finite() && baseline_width > 0.0 {
            format!("{:.2e}x", baseline_width / final_width)
        } else {
            "-".to_string()
        };

        println!(
            "| {:23} | {:9} | {:11.2e} | {:11} |",
            name, elapsed, final_width, improvement
        );
    }

    println!("\n### Interpretation");
    println!(
        "- 'vs Baseline' shows improvement factor compared to IBP + backward LN (conservative)"
    );
    println!("- Forward LN (default) provides ~1e31x improvement - the key optimization");
    println!("- Zonotope provides marginal (~10%) additional tightening at ~20% performance cost");
}
