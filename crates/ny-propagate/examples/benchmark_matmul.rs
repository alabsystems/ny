// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Benchmark comparing ndarray vs faer matrix multiplication
//!
//! Run with: cargo run --release --example benchmark_matmul -p ny-propagate

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

use faer::Mat;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    // Test dimensions matching our IBP benchmark
    let batch = 16;
    let input_dim = 384;
    let output_dim = 1536;
    let iterations = 100;

    println!("Matrix multiply benchmark");
    println!("========================");
    println!(
        "A: [{}, {}]  B: [{}, {}]",
        batch, input_dim, input_dim, output_dim
    );
    println!("Iterations: {}", iterations);
    println!();

    // Create random matrices
    let a_data: Vec<f32> = (0..(batch * input_dim))
        .map(|i| (i as f32 * 0.001) % 1.0)
        .collect();
    let b_data: Vec<f32> = (0..(input_dim * output_dim))
        .map(|i| (i as f32 * 0.001) % 1.0)
        .collect();

    // ndarray version
    let a_nd = Array2::from_shape_vec((batch, input_dim), a_data.clone()).unwrap();
    let b_nd = Array2::from_shape_vec((input_dim, output_dim), b_data.clone()).unwrap();

    // Warmup
    for _ in 0..10 {
        let _ = a_nd.dot(&b_nd);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a_nd.dot(&b_nd);
    }
    let ndarray_time = start.elapsed();
    println!(
        "ndarray dot:  {:?} total, {:.3}ms per iter",
        ndarray_time,
        ndarray_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    // faer version
    let a_faer = Mat::<f32>::from_fn(batch, input_dim, |i, j| a_data[i * input_dim + j]);
    let b_faer = Mat::<f32>::from_fn(input_dim, output_dim, |i, j| b_data[i * output_dim + j]);

    // Warmup
    for _ in 0..10 {
        let _ = &a_faer * &b_faer;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = &a_faer * &b_faer;
    }
    let faer_time = start.elapsed();
    println!(
        "faer matmul:  {:?} total, {:.3}ms per iter",
        faer_time,
        faer_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    println!();
    println!(
        "Speedup: {:.2}x",
        ndarray_time.as_secs_f64() / faer_time.as_secs_f64()
    );
}
