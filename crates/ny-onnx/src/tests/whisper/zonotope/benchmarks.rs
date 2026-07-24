// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::*;
use super::super::helpers::{whisper_tiny_encoder, whisper_zero_input};
use ny_tensor::BoundedTensor;
use std::time::Instant;

/// Run IBP and zonotope through `num_blocks` blocks, collecting timing and width data.
/// Returns `(ibp_times_ms, zono_times_ms, ibp_widths, zono_widths)`.
fn run_timed_blocks(
    whisper: &WhisperModel,
    input: &BoundedTensor,
    num_blocks: usize,
) -> (Vec<u128>, Vec<u128>, Vec<f32>, Vec<f32>) {
    let ibp_config = MultiBlockConfig::default().with_terminate_on_overflow(false);
    let zonotope_config = MultiBlockConfig::tightest_attention().with_terminate_on_overflow(false);
    let mut ibp_current = input.clone();
    let mut zono_current = input.clone();
    let (mut ibp_times, mut zono_times) = (Vec::new(), Vec::new());
    let (mut ibp_widths, mut zono_widths) = (Vec::new(), Vec::new());

    for block in 0..num_blocks {
        let ibp_start = Instant::now();
        let ibp_result = whisper
            .verify_block_compositional_gpu_with_config(block, &ibp_current, None, &ibp_config)
            .unwrap_or_else(|e| panic!("IBP failed at block {block}: {e:?}"));
        ibp_times.push(ibp_start.elapsed().as_millis());

        let zono_start = Instant::now();
        let zono_result = whisper
            .verify_block_compositional_gpu_with_config(
                block,
                &zono_current,
                None,
                &zonotope_config,
            )
            .unwrap_or_else(|e| panic!("Zonotope failed at block {block}: {e:?}"));
        zono_times.push(zono_start.elapsed().as_millis());

        let (iw, zw) = (ibp_result.0.max_width(), zono_result.0.max_width());
        assert!(iw.is_finite(), "IBP bounds not finite at block {block}");
        assert!(
            zw.is_finite(),
            "Zonotope bounds not finite at block {block}"
        );
        ibp_widths.push(iw);
        zono_widths.push(zw);
        ibp_current = ibp_result.0.clone();
        zono_current = zono_result.0.clone();
    }
    (ibp_times, zono_times, ibp_widths, zono_widths)
}

fn print_timing_table(ibp_times: &[u128], zono_times: &[u128], num_blocks: usize) {
    println!("\n### Timing Comparison");
    println!("\n| Block | IBP Time (ms) | Zono Time (ms) | Slowdown |");
    println!("|-------|---------------|----------------|----------|");
    for block in 0..num_blocks {
        let slowdown = if ibp_times[block] > 0 {
            format!("{:.1}x", zono_times[block] as f64 / ibp_times[block] as f64)
        } else {
            "-".to_string()
        };
        println!(
            "| {:5} | {:13} | {:14} | {:8} |",
            block, ibp_times[block], zono_times[block], slowdown
        );
    }
    let total_ibp: u128 = ibp_times.iter().sum();
    let total_zono: u128 = zono_times.iter().sum();
    println!(
        "| Total | {:13} | {:14} | {:8} |",
        total_ibp,
        total_zono,
        format!("{:.1}x", total_zono as f64 / total_ibp as f64)
    );
}

fn print_width_table_and_summary(
    ibp_times: &[u128],
    zono_times: &[u128],
    ibp_widths: &[f32],
    zono_widths: &[f32],
    num_blocks: usize,
) {
    println!("\n### Bound Width Comparison");
    println!("\n| Block | IBP Width | Zonotope Width | Improvement |");
    println!("|-------|-----------|----------------|-------------|");
    for block in 0..num_blocks {
        let ratio = if zono_widths[block] > 0.0 && zono_widths[block].is_finite() {
            format!("{:.2e}x", ibp_widths[block] / zono_widths[block])
        } else {
            "-".to_string()
        };
        println!(
            "| {:5} | {:9.2e} | {:14.2e} | {:11} |",
            block, ibp_widths[block], zono_widths[block], ratio
        );
    }

    let total_ibp: u128 = ibp_times.iter().sum();
    let total_zono: u128 = zono_times.iter().sum();
    let final_ibp = ibp_widths.last().unwrap_or(&f32::INFINITY);
    let final_zono = zono_widths.last().unwrap_or(&f32::INFINITY);
    println!("\n### Summary");
    println!("- Total IBP time: {} ms", total_ibp);
    println!("- Total Zonotope time: {} ms", total_zono);
    println!(
        "- Zonotope slowdown: {:.1}x",
        total_zono as f64 / total_ibp as f64
    );
    println!("- Final IBP width: {:.2e}", final_ibp);
    println!("- Final Zonotope width: {:.2e}", final_zono);
    if final_zono.is_finite() && *final_zono > 0.0 {
        println!(
            "- Final bound improvement: {:.2e}x tighter",
            final_ibp / final_zono
        );
    }

    let final_ratio =
        if zono_widths[num_blocks - 1].is_finite() && zono_widths[num_blocks - 1] > 0.0 {
            ibp_widths[num_blocks - 1] / zono_widths[num_blocks - 1]
        } else {
            0.0
        };
    assert!(
        final_ratio > 0.5,
        "Zonotope bounds should be within 2x of IBP (ratio {:.2e})",
        final_ratio
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_zonotope_performance_benchmark() {
    // Comprehensive benchmark comparing zonotope vs IBP: tightness + timing
    let whisper = whisper_tiny_encoder();
    assert_eq!(whisper.encoder_layers, 4, "Expected 4 encoder layers");
    let num_blocks = whisper.encoder_layers;
    let input = whisper_zero_input(whisper.hidden_dim, 8, 0.001);

    println!("\n=== Zonotope vs IBP Performance Benchmark ===");
    println!(
        "Model: Whisper-tiny, {} blocks, seq=8, hidden={}, eps=0.001",
        num_blocks, whisper.hidden_dim
    );

    let (ibp_times, zono_times, ibp_widths, zono_widths) =
        run_timed_blocks(whisper, &input, num_blocks);
    print_timing_table(&ibp_times, &zono_times, num_blocks);
    print_width_table_and_summary(
        &ibp_times,
        &zono_times,
        &ibp_widths,
        &zono_widths,
        num_blocks,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_zonotope_scaling_benchmark() {
    // Benchmark zonotope performance scaling with sequence length
    let whisper = whisper_tiny_encoder();
    let hidden_dim = whisper.hidden_dim;
    let epsilon = 0.001;

    println!("\n=== Zonotope Scaling Benchmark ===");
    println!("Model: Whisper-tiny (single block 0), eps={}", epsilon);
    println!(
        "\n| Seq Len | IBP (ms) | Zono (ms) | Slowdown | IBP Width | Zono Width | Improvement |"
    );
    println!(
        "|---------|----------|-----------|----------|-----------|------------|-------------|"
    );

    let zonotope_config = MultiBlockConfig::tightest_attention().with_terminate_on_overflow(false);
    let ibp_config = MultiBlockConfig::default().with_terminate_on_overflow(false);

    for &seq_len in &[8, 16, 32, 64] {
        let input = whisper_zero_input(hidden_dim, seq_len, epsilon);

        let ibp_start = Instant::now();
        let ibp_result = whisper
            .verify_block_compositional_gpu_with_config(0, &input, None, &ibp_config)
            .unwrap_or_else(|e| panic!("IBP compositional failed: {:?}", e));
        let ibp_elapsed = ibp_start.elapsed().as_millis();

        let zono_start = Instant::now();
        let zono_result = whisper
            .verify_block_compositional_gpu_with_config(0, &input, None, &zonotope_config)
            .unwrap_or_else(|e| panic!("Zonotope compositional failed: {:?}", e));
        let zono_elapsed = zono_start.elapsed().as_millis();

        let (ibp_width, zono_width) = (ibp_result.0.max_width(), zono_result.0.max_width());
        let slowdown = if ibp_elapsed > 0 {
            format!("{:.1}x", zono_elapsed as f64 / ibp_elapsed as f64)
        } else {
            "-".to_string()
        };
        let improvement = if zono_width > 0.0 && zono_width.is_finite() {
            format!("{:.0}x", ibp_width / zono_width)
        } else {
            "-".to_string()
        };

        println!(
            "| {:7} | {:8} | {:9} | {:8} | {:9.2e} | {:10.2e} | {:11} |",
            seq_len, ibp_elapsed, zono_elapsed, slowdown, ibp_width, zono_width, improvement
        );
    }
}
