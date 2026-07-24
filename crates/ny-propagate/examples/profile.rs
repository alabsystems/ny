// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Profiling benchmark for ny operations
//!
//! Run with: cargo build --release -p ny-propagate --example profile && \
//!           ./target/release/examples/profile

use ndarray::{Array2, ArrayD, IxDyn};
use ny_gpu::{AcceleratedBoundPropagation, AcceleratedDevice};
use ny_propagate::layers::{GELULayer, LayerNormLayer, LinearLayer, MatMulLayer, SoftmaxLayer};
use ny_propagate::{BoundPropagation, GraphNetwork, GraphNode, Layer, Network};
use ny_tensor::BoundedTensor;
use std::time::Instant;

/// Simple timing helper
fn bench<F: FnMut()>(name: &str, iterations: usize, mut f: F) {
    // Warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;

    println!(
        "{}: {:?} per iteration ({} iterations)",
        name, per_iter, iterations
    );
}

/// Create a BoundedTensor with specified shape from center and epsilon
fn make_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon).unwrap()
}

fn main() {
    println!("ny Profiling Benchmark");
    println!(
        "===========================
"
    );

    // Whisper-tiny dimensions
    let batch = 1;
    let seq_len = 4; // Small for now, Whisper can have up to 1500
    let hidden_dim = 384;
    let intermediate_dim = 1536;
    let num_heads = 6;
    let head_dim = 64; // 384 / 6

    println!("Dimensions:");
    println!(
        "  batch={}, seq={}, hidden={}, intermediate={}",
        batch, seq_len, hidden_dim, intermediate_dim
    );
    println!(
        "  heads={}, head_dim={}
",
        num_heads, head_dim
    );

    let epsilon = 0.01_f32;

    // Create test input
    let input = make_input(&[batch, seq_len, hidden_dim], 0.5, epsilon);

    // Linear layer: [batch, seq, hidden] -> [batch, seq, intermediate]
    // Weight shape is [out_features, in_features] = [1536, 384]
    let linear_weight = Array2::from_shape_fn((intermediate_dim, hidden_dim), |_| 0.01_f32);
    let linear_bias = Some(ndarray::Array1::zeros(intermediate_dim));
    let linear1 = LinearLayer::new(linear_weight.clone(), linear_bias.clone()).unwrap();

    // Linear layer back: [batch, seq, intermediate] -> [batch, seq, hidden]
    // Weight shape is [out_features, in_features] = [384, 1536]
    let linear_weight2 = Array2::from_shape_fn((hidden_dim, intermediate_dim), |_| 0.01_f32);
    let linear_bias2 = Some(ndarray::Array1::zeros(hidden_dim));
    let linear2 = LinearLayer::new(linear_weight2, linear_bias2).unwrap();

    // GELU
    let gelu = GELULayer::default();

    // LayerNorm
    let layernorm = LayerNormLayer::new(
        ndarray::Array1::ones(hidden_dim),
        ndarray::Array1::zeros(hidden_dim),
        1e-5,
    )
    .unwrap();

    println!(
        "=== IBP Benchmarks ===
"
    );

    // Benchmark: Linear layer IBP
    let mut linear_output = input.clone();
    bench("Linear IBP [384->1536]", 100, || {
        linear_output = linear1.propagate_ibp(&input).unwrap();
    });

    // Benchmark: GELU IBP
    bench("GELU IBP [1536]", 100, || {
        let _ = gelu.propagate_ibp(&linear_output);
    });
    let gelu_output = gelu.propagate_ibp(&linear_output).unwrap();

    // Benchmark: Linear layer 2 IBP
    bench("Linear IBP [1536->384]", 100, || {
        let _ = linear2.propagate_ibp(&gelu_output);
    });
    let final_output = linear2.propagate_ibp(&gelu_output).unwrap();

    // Benchmark: LayerNorm IBP
    bench("LayerNorm IBP [384]", 100, || {
        let _ = layernorm.propagate_ibp(&final_output);
    });

    println!(
        "
=== Full MLP Path IBP ===
"
    );

    // Build MLP network
    let mut mlp = Network::new();
    mlp.add_layer(Layer::Linear(linear1.clone()));
    mlp.add_layer(Layer::GELU(gelu.clone()));
    mlp.add_layer(Layer::Linear(linear2.clone()));

    bench("Full MLP IBP [384->1536->384]", 100, || {
        let _ = mlp.propagate_ibp(&input);
    });

    println!(
        "
=== CROWN Benchmarks ===
"
    );

    // CROWN needs per-position execution for N-D
    // For now, test 1-D version
    let input_1d = make_input(&[hidden_dim], 0.5, epsilon);

    bench("Full MLP CROWN 1-D [384]", 100, || {
        let _ = mlp.propagate_crown(&input_1d);
    });

    println!(
        "
=== Attention Components ===
"
    );

    // MatMul benchmark: Q @ K^T
    let q_input = make_input(&[batch, num_heads, seq_len, head_dim], 0.5, 0.1);
    let k_input = make_input(&[batch, num_heads, head_dim, seq_len], 0.5, 0.1);

    let matmul = MatMulLayer::new(false, None);

    bench("MatMul IBP [1,6,4,64] @ [1,6,64,4]", 100, || {
        let _ = matmul.propagate_ibp_binary(&q_input, &k_input);
    });

    // Softmax benchmark
    let attn_input = make_input(&[batch, num_heads, seq_len, seq_len], 0.0, 1.0);

    let softmax = SoftmaxLayer::new(-1);

    bench("Softmax IBP [1,6,4,4]", 100, || {
        let _ = softmax.propagate_ibp(&attn_input);
    });

    println!(
        "
=== Scaling Tests ===
"
    );

    // Test with larger sequence length
    for seq in [4, 16, 64, 128] {
        let large_input = make_input(&[batch, seq, hidden_dim], 0.5, epsilon);

        let iterations = if seq <= 16 {
            100
        } else if seq <= 64 {
            20
        } else {
            5
        };
        bench(&format!("MLP IBP seq={}", seq), iterations, || {
            let _ = mlp.propagate_ibp(&large_input);
        });
    }

    println!(
        "
=== Per-Position CROWN (N-D) ===
"
    );

    // Test per-position CROWN on a GraphNetwork
    let mut graph = GraphNetwork::new();

    // Build MLP graph
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(linear1.clone()),
    ));
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

    bench("Per-Position CROWN MLP [4,384]", 20, || {
        let _ = graph.propagate_crown_per_position(&input);
    });

    // Larger sequences
    for seq in [16, 64] {
        let large_input = make_input(&[batch, seq, hidden_dim], 0.5, epsilon);

        let iterations = if seq <= 16 { 10 } else { 3 };
        bench(
            &format!("Per-Position CROWN seq={}", seq),
            iterations,
            || {
                let _ = graph.propagate_crown_per_position(&large_input);
            },
        );
    }

    println!(
        "
=== ny-gpu Accelerated Comparison ===
"
    );

    // Compare accelerated vs existing implementation
    let accel_device = AcceleratedDevice::new();

    // Test accelerated Linear IBP
    bench("ny-gpu Linear IBP [384->1536]", 100, || {
        let _ = accel_device.linear_ibp(&input, &linear_weight, linear_bias.as_ref());
    });

    // Test accelerated MatMul IBP
    bench("ny-gpu MatMul IBP [1,6,4,64]@[1,6,64,4]", 100, || {
        let _ = accel_device.matmul_ibp(&q_input, &k_input);
    });

    // Scaling comparison
    println!(
        "
=== Scaling Comparison (existing vs ny-gpu) ===
"
    );

    for seq in [4, 16, 64, 128] {
        let large_input = make_input(&[batch, seq, hidden_dim], 0.5, epsilon);
        let iterations = if seq <= 16 {
            100
        } else if seq <= 64 {
            20
        } else {
            5
        };

        // Existing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = linear1.propagate_ibp(&large_input);
        }
        let existing_per_iter = start.elapsed() / iterations as u32;

        // Accelerated
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = accel_device.linear_ibp(&large_input, &linear_weight, linear_bias.as_ref());
        }
        let accel_per_iter = start.elapsed() / iterations as u32;

        let speedup = existing_per_iter.as_nanos() as f64 / accel_per_iter.as_nanos() as f64;
        println!(
            "Linear IBP seq={}: existing={:?}, accel={:?}, speedup={:.2}x",
            seq, existing_per_iter, accel_per_iter, speedup
        );
    }

    println!(
        "
=== Parallel Per-Position CROWN Comparison ===
"
    );

    for seq in [4, 16, 64] {
        let large_input = make_input(&[batch, seq, hidden_dim], 0.5, epsilon);
        let iterations = if seq <= 4 {
            10
        } else if seq <= 16 {
            5
        } else {
            2
        };

        // Sequential per-position CROWN
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = graph.propagate_crown_per_position(&large_input);
        }
        let seq_per_iter = start.elapsed() / iterations as u32;

        // Parallel per-position CROWN
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = accel_device.crown_per_position_parallel(&graph, &large_input);
        }
        let par_per_iter = start.elapsed() / iterations as u32;

        let speedup = seq_per_iter.as_nanos() as f64 / par_per_iter.as_nanos() as f64;
        println!(
            "Per-Position CROWN seq={}: sequential={:?}, parallel={:?}, speedup={:.2}x",
            seq, seq_per_iter, par_per_iter, speedup
        );
    }

    println!(
        "
=== Summary ===
"
    );
    println!("Profile complete. Key findings:");
    println!("- Linear layers are the main bottleneck (O(n*m) operations)");
    println!("- Per-position CROWN scales linearly with sequence length");
    println!("- ny-gpu provides parallel acceleration via Rayon");
}
