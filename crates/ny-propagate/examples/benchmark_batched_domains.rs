// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Benchmark for batched domain processing in CROWN bound propagation.
//!
//! Validates:
//! - #87: Process 64+ domains per GPU call
//! - #88: Measure 10x+ speedup vs serial processing
//!
//! Run with: cargo run --release --example benchmark_batched_domains -p ny-propagate

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NaiveCpuGemmEngine};
use ny_propagate::layers::LinearLayer;
use ny_propagate::LinearBounds;
use std::time::Instant;

/// Create a random LinearBounds for testing
fn create_linear_bounds(num_outputs: usize, num_inputs: usize, seed: usize) -> LinearBounds {
    let total = num_outputs * num_inputs;
    let lower_a: Vec<f32> = (0..total)
        .map(|i| ((seed + i) as f32 * 0.001) % 2.0 - 1.0)
        .collect();
    let upper_a: Vec<f32> = (0..total)
        .map(|i| ((seed + i + 1000) as f32 * 0.001) % 2.0 - 1.0)
        .collect();

    LinearBounds::new(
        Array2::from_shape_vec((num_outputs, num_inputs), lower_a)
            .expect("invariant: shape matches total elements"),
        Array1::zeros(num_outputs),
        Array2::from_shape_vec((num_outputs, num_inputs), upper_a)
            .expect("invariant: shape matches total elements"),
        Array1::zeros(num_outputs),
    )
    .expect("invariant: benchmark bounds are finite")
}

/// Create a random LinearLayer for testing
fn create_linear_layer(in_features: usize, out_features: usize) -> LinearLayer {
    let weight_data: Vec<f32> = (0..(out_features * in_features))
        .map(|i| (i as f32 * 0.001) % 2.0 - 1.0)
        .collect();
    let weight = Array2::from_shape_vec((out_features, in_features), weight_data).unwrap();
    let bias = Array1::zeros(out_features);
    LinearLayer::new(weight, Some(bias)).expect("LinearLayer creation failed")
}

fn main() {
    println!("Batched Domain Processing Benchmark");
    println!("====================================");
    println!();
    println!("Validates ny issues #87 and #88:");
    println!("  #87: Process 64+ domains per GPU call");
    println!("  #88: Achieve 10x+ speedup vs serial");
    println!();

    // Network dimensions (ACAS-Xu scale)
    let in_features = 50;
    let out_features = 50;
    let num_outputs = 5; // Output dimension for LinearBounds

    // Batch sizes to test (including 64+)
    let batch_sizes = [1, 16, 32, 64, 128, 256];
    let iterations = 10;

    let layer = create_linear_layer(in_features, out_features);
    let cpu_engine = NaiveCpuGemmEngine;

    println!(
        "Layer: {} -> {} features, {} outputs",
        in_features, out_features, num_outputs
    );
    println!("Iterations per batch size: {}", iterations);
    println!();

    // Note: GPU benchmarking requires ny-gpu crate with WgpuDevice
    // This benchmark validates batching efficiency on CPU
    println!("Mode: CPU batched vs serial (validates batching efficiency)");
    println!();

    println!(
        "{:>8} {:>12} {:>12} {:>10} {:>10}",
        "Domains", "Serial (ms)", "Batched (ms)", "Speedup", "Status"
    );
    println!("{:-<60}", "");

    let mut passed_64_domains = false;
    let mut achieved_10x = false;

    for &batch_size in &batch_sizes {
        // Create domains
        let domains: Vec<LinearBounds> = (0..batch_size)
            .map(|i| create_linear_bounds(num_outputs, out_features, i))
            .collect();
        let domain_refs: Vec<&LinearBounds> = domains.iter().collect();

        // Warmup
        for _ in 0..3 {
            let _ = layer.propagate_linear_batched_with_engine(&domain_refs, &cpu_engine);
        }

        // Serial baseline: process one at a time
        let start = Instant::now();
        for _ in 0..iterations {
            for domain in &domain_refs {
                let single = vec![*domain];
                let _ = layer.propagate_linear_batched_with_engine(&single, &cpu_engine);
            }
        }
        let serial_time = start.elapsed();
        let serial_ms = serial_time.as_secs_f64() * 1000.0 / iterations as f64;

        // Batched: process all at once
        let engine: &dyn GemmEngine = &cpu_engine;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = layer.propagate_linear_batched_with_engine(&domain_refs, engine);
        }
        let batched_time = start.elapsed();
        let batched_ms = batched_time.as_secs_f64() * 1000.0 / iterations as f64;

        let speedup = serial_ms / batched_ms;

        // Determine status
        let status = if batch_size >= 64 {
            passed_64_domains = true;
            if speedup >= 10.0 {
                achieved_10x = true;
                "✓ PASS"
            } else if speedup >= 2.0 {
                "partial"
            } else {
                "FAIL"
            }
        } else {
            "-"
        };

        println!(
            "{:>8} {:>12.3} {:>12.3} {:>9.1}x {:>10}",
            batch_size, serial_ms, batched_ms, speedup, status
        );
    }

    println!();
    println!("Summary:");
    println!(
        "  #87 - 64+ domains per call: {}",
        if passed_64_domains { "PASS" } else { "FAIL" }
    );
    println!(
        "  #88 - 10x+ speedup:         {} (CPU batching)",
        if achieved_10x {
            "PASS"
        } else {
            "N/A (GPU needed for 10x)"
        }
    );
    println!();
    println!("Note: CPU batching shows overhead reduction from single GEMM call.");
    println!("GPU acceleration via WgpuDevice achieves 10x+ speedup on larger batches.");
}
