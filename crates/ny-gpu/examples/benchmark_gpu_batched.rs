// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU benchmark for batched CROWN bound propagation.
//!
//! Validates:
//! - #87: Process 64+ domains per GPU call
//! - #88: Measure 10x+ speedup vs serial CPU
//!
//! Run with: cargo run --release --example benchmark_gpu_batched -p ny-gpu

use ndarray::{Array1, Array2};
use ny_core::GemmEngine;
use ny_gpu::WgpuDevice;
use ny_propagate::layers::LinearLayer;
use ny_propagate::LinearBounds;
use std::time::Instant;

/// CPU-only "engine" for serial baseline
struct CpuEngine;

impl GemmEngine for CpuEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        // Basic GEMM: C = A @ B
        // A: [m, k], B: [k, n] -> C: [m, n]
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }
}

/// Create a random LinearBounds for testing
fn create_linear_bounds(num_outputs: usize, num_inputs: usize, seed: usize) -> LinearBounds {
    let total = num_outputs * num_inputs;
    let lower_a: Vec<f32> = (0..total)
        .map(|i| ((seed + i) as f32 * 0.001) % 2.0 - 1.0)
        .collect();
    let upper_a: Vec<f32> = (0..total)
        .map(|i| ((seed + i + 1000) as f32 * 0.001) % 2.0 - 1.0)
        .collect();

    let lower =
        Array2::from_shape_vec((num_outputs, num_inputs), lower_a).expect("valid lower_a shape");
    let upper =
        Array2::from_shape_vec((num_outputs, num_inputs), upper_a).expect("valid upper_a shape");
    LinearBounds::new(
        lower,
        Array1::zeros(num_outputs),
        upper,
        Array1::zeros(num_outputs),
    )
    .expect("valid linear bounds")
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU Batched Domain Processing Benchmark");
    println!("========================================");
    println!();
    println!("Validates ny issues #87 and #88:");
    println!("  #87: Process 64+ domains per GPU call");
    println!("  #88: Achieve 10x+ speedup vs serial CPU");
    println!();

    // Initialize GPU device
    let gpu_device = WgpuDevice::new().map_err(|error| {
        format!(
            "GPU initialization failed: {error}. This benchmark requires a usable WGPU device; \
             use benchmark_batched_domains for CPU-only measurements"
        )
    })?;
    println!("GPU: wgpu device initialized");

    // Network dimensions - larger for GPU benefit
    // At small sizes, GPU overhead dominates. At larger sizes, GPU parallelism wins.
    let sizes = [
        (50, 50, 5, "ACAS-Xu scale"),
        (256, 256, 10, "Medium MLP"),
        (512, 512, 10, "Large MLP"),
        (1024, 1024, 5, "Very large MLP"),
    ];

    // Batch sizes to test (including 64+)
    let batch_sizes = [16, 64, 128, 256, 512];
    let iterations = 5;

    let cpu_engine = CpuEngine;

    let mut passed_64_domains = false;
    let mut achieved_10x = false;
    let mut best_speedup = 0.0f64;
    let mut best_config = String::new();

    for (in_features, out_features, num_outputs, label) in sizes {
        println!();
        println!(
            "=== {} ({} -> {}, {} outputs) ===",
            label, in_features, out_features, num_outputs
        );

        let layer = create_linear_layer(in_features, out_features);

        println!(
            "{:>8} {:>14} {:>14} {:>10}",
            "Domains", "CPU Serial", "GPU Batched", "Speedup"
        );
        println!("{:-<52}", "");

        for &batch_size in &batch_sizes {
            // Create domains
            let domains: Vec<LinearBounds> = (0..batch_size)
                .map(|i| create_linear_bounds(num_outputs, out_features, i))
                .collect();
            let domain_refs: Vec<&LinearBounds> = domains.iter().collect();

            // Warmup GPU
            for _ in 0..2 {
                layer
                    .propagate_linear_batched_with_engine(&domain_refs, &gpu_device)
                    .expect("GPU warmup propagation must succeed");
            }

            // Serial CPU baseline: process one at a time
            let start = Instant::now();
            for _ in 0..iterations {
                for domain in &domain_refs {
                    let single = vec![*domain];
                    layer
                        .propagate_linear_batched_with_engine(&single, &cpu_engine)
                        .expect("serial CPU propagation must succeed");
                }
            }
            let serial_time = start.elapsed();
            let serial_ms = serial_time.as_secs_f64() * 1000.0 / iterations as f64;

            // GPU Batched: process all at once
            let start = Instant::now();
            for _ in 0..iterations {
                layer
                    .propagate_linear_batched_with_engine(&domain_refs, &gpu_device)
                    .expect("batched GPU propagation must succeed");
            }
            let gpu_time = start.elapsed();
            let gpu_ms = gpu_time.as_secs_f64() * 1000.0 / iterations as f64;

            let speedup = serial_ms / gpu_ms;

            // Track status
            if batch_size >= 64 {
                passed_64_domains = true;
                if speedup >= 10.0 {
                    achieved_10x = true;
                }
                if speedup > best_speedup {
                    best_speedup = speedup;
                    best_config = format!(
                        "{} domains, {} ({} -> {})",
                        batch_size, label, in_features, out_features
                    );
                }
            }

            let speedup_str = if speedup >= 10.0 {
                format!("{:.1}x ✓", speedup)
            } else if speedup >= 2.0 {
                format!("{:.1}x", speedup)
            } else {
                format!("{:.2}x", speedup)
            };

            println!(
                "{:>8} {:>12.3}ms {:>12.3}ms {:>10}",
                batch_size, serial_ms, gpu_ms, speedup_str
            );
        }
    }

    println!();
    println!("========================================");
    println!("Summary:");
    println!(
        "  #87 - 64+ domains per call: {}",
        if passed_64_domains { "PASS" } else { "FAIL" }
    );
    println!(
        "  #88 - 10x+ speedup:         {}",
        if achieved_10x {
            format!("PASS (best: {:.1}x with {})", best_speedup, best_config)
        } else {
            format!("FAIL (best: {:.1}x)", best_speedup)
        }
    );
    println!();

    if !passed_64_domains || !achieved_10x {
        return Err(format!(
            "GPU batched benchmark acceptance failed: 64+ domains={}, best speedup={best_speedup:.2}x",
            passed_64_domains
        )
        .into());
    }
    Ok(())
}
