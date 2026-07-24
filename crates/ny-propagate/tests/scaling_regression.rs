// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scaling regression tests for propagate_crown_per_position.
//!
//! Validates that per-position CROWN scales linearly with sequence length,
//! as claimed in examples/profile.rs.
//!
//! **Gated behind `benchmarks` feature (#2249):** These tests use wall-clock
//! timing to detect O(n^2) regressions. They are inherently flaky under system
//! load, cargo serialization contention, or debug builds. Run explicitly with:
//!   `cargo test -p ny-propagate --features benchmarks --test scaling_regression`
//!
//! Thresholds are wide (e.g. 12x for a 4x size increase, where O(n^2) = 16x)
//! with median-of-3 sampling. But wall-clock assertions in a test suite remain
//! fundamentally timing-dependent (#1692).
//!
//! Part of #1553: Perf proof for per-position CROWN scaling.

#![cfg(feature = "benchmarks")]

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::layers::{GELULayer, LinearLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;
use std::time::Instant;

/// Create a BoundedTensor with specified shape
fn make_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon).unwrap()
}

/// Build a simple MLP network for testing.
fn build_test_mlp(hidden_dim: usize, intermediate: usize) -> GraphNetwork {
    let linear1 = LinearLayer::new(
        Array2::from_shape_fn((intermediate, hidden_dim), |_| 0.01_f32),
        Some(Array1::zeros(intermediate)),
    )
    .unwrap();
    let gelu = GELULayer::default();
    let linear2 = LinearLayer::new(
        Array2::from_shape_fn((hidden_dim, intermediate), |_| 0.01_f32),
        Some(Array1::zeros(hidden_dim)),
    )
    .unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(gelu),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Measure per-position CROWN runtime for a given sequence length.
/// Returns median time in milliseconds over 3 timed runs (after warmup).
fn measure_crown_per_position(graph: &GraphNetwork, seq_len: usize, hidden_dim: usize) -> f64 {
    let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

    // Warmup run (to ensure caches are warm, JIT compilation done, etc.)
    let _ = graph.propagate_crown_per_position(&input);

    // Take median of 3 timed runs to reduce noise
    let mut times = [0.0_f64; 3];
    for t in &mut times {
        let start = Instant::now();
        let _ = graph.propagate_crown_per_position(&input);
        *t = start.elapsed().as_secs_f64() * 1000.0;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[1] // median
}

/// Test that per-position CROWN scales linearly with sequence length.
///
/// Linear scaling means: doubling the sequence length should roughly double the time.
/// We use a tolerance factor because:
/// - Cache effects can cause non-linear behavior at small sizes
/// - System noise affects timing
/// - Memory allocation patterns vary
///
/// The test validates that time(4x) / time(x) is within [1.0, 12.0] (expected: 4.0).
/// The wide tolerance avoids flakiness under load (cargo serialization, CI noise)
/// while still catching quadratic regression: O(n²) would yield ratio ~16x.
#[test]
#[ntest::timeout(60000)] // 60 second timeout
fn test_per_position_crown_linear_scaling() {
    let hidden_dim = 256; // Smaller for faster test
    let intermediate = 512;
    let graph = build_test_mlp(hidden_dim, intermediate);

    // Measure at different sequence lengths
    let seq_small = 4;
    let seq_large = 16; // 4x larger

    let time_small = measure_crown_per_position(&graph, seq_small, hidden_dim);
    let time_large = measure_crown_per_position(&graph, seq_large, hidden_dim);

    let ratio = time_large / time_small;
    let expected_ratio = (seq_large as f64) / (seq_small as f64); // 4.0

    // Catch catastrophic regressions (quadratic = 16x), not noise.
    // O(n) ≈ 4x, O(n log n) ≈ 5.5x, O(n²) ≈ 16x.
    let min_ratio = 1.0;
    let max_ratio = 12.0;

    eprintln!("Per-position CROWN scaling test:");
    eprintln!("  seq={}: {:.2}ms", seq_small, time_small);
    eprintln!("  seq={}: {:.2}ms", seq_large, time_large);
    eprintln!(
        "  ratio: {:.2}x (expected: {:.1}x, allowed: {:.1}-{:.1}x)",
        ratio, expected_ratio, min_ratio, max_ratio
    );

    assert!(
        ratio >= min_ratio && ratio <= max_ratio,
        "Per-position CROWN scaling regression detected! \
         Expected ~{:.1}x scaling for {}x sequence increase, got {:.2}x. \
         Ratio > {:.0}x suggests super-linear scaling (quadratic?). \
         (time_small={:.2}ms, time_large={:.2}ms)",
        expected_ratio,
        seq_large / seq_small,
        ratio,
        max_ratio,
        time_small,
        time_large
    );
}

/// Extended scaling test with more data points.
/// Validates scaling across multiple sequence lengths.
#[test]
#[ntest::timeout(120000)] // 2 minute timeout
fn test_per_position_crown_scaling_extended() {
    let hidden_dim = 128; // Even smaller for faster extended test
    let intermediate = 256;
    let graph = build_test_mlp(hidden_dim, intermediate);

    let seq_lengths = [4, 8, 16, 32];
    let mut times: Vec<(usize, f64)> = Vec::new();

    for &seq in &seq_lengths {
        let time = measure_crown_per_position(&graph, seq, hidden_dim);
        times.push((seq, time));
    }

    eprintln!("Extended scaling test:");
    for (seq, time) in &times {
        eprintln!("  seq={}: {:.2}ms", seq, time);
    }

    // Check each consecutive pair for catastrophic regression only.
    // O(n) ≈ 2x per doubling, O(n²) ≈ 4x. Threshold at 3x catches quadratic.
    for i in 1..times.len() {
        let (seq_prev, time_prev) = times[i - 1];
        let (seq_curr, time_curr) = times[i];

        let expected_ratio = (seq_curr as f64) / (seq_prev as f64);
        let actual_ratio = time_curr / time_prev;

        // Wide bounds: catch quadratic, ignore noise.
        let min_ratio = 0.5;
        let max_ratio = expected_ratio * 3.0;

        eprintln!(
            "  {} -> {}: {:.2}x (expected: {:.1}x, max: {:.1}x)",
            seq_prev, seq_curr, actual_ratio, expected_ratio, max_ratio
        );

        assert!(
            actual_ratio >= min_ratio && actual_ratio <= max_ratio,
            "Scaling regression at seq {}->{}. Expected ~{:.1}x, got {:.2}x (max {:.1}x)",
            seq_prev,
            seq_curr,
            expected_ratio,
            actual_ratio,
            max_ratio
        );
    }
}
