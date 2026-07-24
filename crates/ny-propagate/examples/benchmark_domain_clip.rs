// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Benchmark for domain clipping operations
//!
//! Run with: cargo run --release --example benchmark_domain_clip -p ny-propagate

use ndarray::{ArrayD, IxDyn};
use ny_propagate::domain_clip::{ClipStrategy, DomainClipConfig, DomainClipper};
use ny_tensor::BoundedTensor;
use std::time::Instant;

fn main() {
    println!("Domain Clip Benchmark");
    println!("=====================");
    println!("Testing clip_bounds performance at different tensor sizes");
    println!("Parallel threshold: 1,000,000 elements");
    println!();

    let sizes = [
        (64, 384),   // 24,576 elements - below threshold (sequential)
        (64, 1024),  // 65,536 elements - at threshold
        (64, 1536),  // 98,304 elements - above threshold (parallel)
        (256, 1024), // 262,144 elements - above threshold (parallel)
        (256, 4096), // 1,048,576 elements - large parallel
        (512, 4096), // 2,097,152 elements - very large parallel
    ];

    let iterations = 100;
    println!("Iterations per size: {}", iterations);
    println!();

    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>10}",
        "Batch", "Dim", "Elements", "Time/iter", "Mode"
    );
    println!("{:-<60}", "");

    for (batch, dim) in sizes {
        let shape = vec![batch, dim];
        let elements = batch * dim;
        let mode = if elements >= 1_000_000 {
            "parallel"
        } else {
            "sequential"
        };

        // Create bounds
        let lower = ArrayD::from_elem(IxDyn(&shape), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&shape), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        // Create clipper with statistics
        let mut clipper = DomainClipper::new(DomainClipConfig {
            strategy: ClipStrategy::Combined {
                statistical_k: 6.0,
                empirical_margin: 0.1,
            },
            min_samples: 1,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 100.0,
        });

        // Observe values to populate statistics
        for i in 0..20 {
            let val = (i as f32 - 10.0) / 20.0;
            let sample = ArrayD::from_elem(IxDyn(&shape), val);
            clipper.observe("test_layer", &sample).unwrap();
        }

        // Warmup
        for _ in 0..10 {
            let _ = clipper.clip_bounds("test_layer", &bounds).unwrap();
        }

        // Benchmark clip_bounds (includes combined_bounds + ensure_valid_bounds)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = clipper.clip_bounds("test_layer", &bounds).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "{:>8} {:>10} {:>12} {:>10.1}µs {:>10}",
            batch, dim, elements, per_iter_us, mode
        );
    }

    println!();
    println!("Note: 'parallel' mode uses Rayon parallel iterators when elements >= 1,000,000");
    println!("Domain clip operations are memory-bound; parallelization overhead exceeds benefit");
    println!("for smaller tensors.");
}
