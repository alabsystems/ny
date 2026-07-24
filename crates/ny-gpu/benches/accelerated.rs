// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for accelerated bound propagation (ny-gpu)
//!
//! Compares CPU baseline vs wgpu GPU implementations.
//!
//! Run with: cargo bench -p ny-gpu
//! HTML reports: target/criterion/report/index.html
//!
//! ## Isolated Kernel Benchmarks
//!
//! The `GpuKernels` group provides isolated microbenchmarks for individual GPU operations:
//! - Linear IBP (matrix-vector with interval arithmetic)
//! - MatMul IBP (batched matrix multiplication)
//! - Attention IBP (full attention: matmul + softmax + matmul)
//!
//! These help identify which operations benefit most from GPU acceleration.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_gpu::{AcceleratedBoundPropagation, AcceleratedDevice, WgpuDevice};
use ny_propagate::layers::{GELULayer, LinearLayer, MatMulLayer};
use ny_propagate::{BoundPropagation, GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// Create a BoundedTensor with specified shape
fn make_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon).expect("valid bounded tensor")
}

// ============================================================================
// CPU vs Accelerated Comparison
// ============================================================================

fn bench_linear_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison/Linear");

    let device = AcceleratedDevice::new();

    for (in_dim, out_dim) in [(384, 1536), (768, 3072)] {
        let weight = Array2::from_shape_fn((out_dim, in_dim), |_| 0.01_f32);
        let bias = Some(Array1::zeros(out_dim));
        let layer = LinearLayer::new(weight.clone(), bias.clone()).unwrap();

        for seq_len in [64, 256] {
            let input = make_input(&[1, seq_len, in_dim], 0.5, 0.01);
            let id = format!("{}x{}_seq{}", in_dim, out_dim, seq_len);

            group.throughput(Throughput::Elements((seq_len * in_dim * out_dim) as u64));

            // CPU baseline
            group.bench_with_input(
                BenchmarkId::new("cpu", &id),
                &(&layer, &input),
                |b, (layer, input)| b.iter(|| layer.propagate_ibp(black_box(input))),
            );

            // Accelerated (Rayon parallel)
            group.bench_with_input(
                BenchmarkId::new("accel", &id),
                &(&input, &weight, &bias),
                |b, (input, weight, bias)| {
                    b.iter(|| device.linear_ibp(black_box(input), black_box(weight), bias.as_ref()))
                },
            );
        }
    }
    group.finish();
}

fn bench_matmul_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison/MatMul");

    let device = AcceleratedDevice::new();
    let matmul = MatMulLayer::new(false, None);

    for (heads, seq, head_dim) in [(6, 64, 64), (8, 128, 64), (12, 256, 64)] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k = make_input(&[1, heads, head_dim, seq], 0.5, 0.1);
        let id = format!("h{}s{}d{}", heads, seq, head_dim);

        group.throughput(Throughput::Elements((heads * seq * seq * head_dim) as u64));

        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu", &id),
            &(&matmul, &q, &k),
            |b, (matmul, q, k)| b.iter(|| matmul.propagate_ibp_binary(black_box(q), black_box(k))),
        );

        // Accelerated
        group.bench_with_input(BenchmarkId::new("accel", &id), &(&q, &k), |b, (q, k)| {
            b.iter(|| device.matmul_ibp(black_box(q), black_box(k)))
        });
    }
    group.finish();
}

fn bench_attention_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison/Attention");
    group.sample_size(20);

    let device = AcceleratedDevice::new();

    for (heads, seq, head_dim) in [(6, 64, 64), (8, 128, 64)] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let v = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let id = format!("h{}s{}d{}", heads, seq, head_dim);

        group.throughput(Throughput::Elements((heads * seq * seq) as u64));

        // Accelerated attention (includes softmax)
        group.bench_with_input(
            BenchmarkId::new("full", &id),
            &(&q, &k, &v),
            |b, (q, k, v)| {
                b.iter(|| device.attention_ibp(black_box(q), black_box(k), black_box(v), scale))
            },
        );
    }
    group.finish();
}

fn bench_causal_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("CausalAttention");
    group.sample_size(20);

    let device = AcceleratedDevice::new();

    for (heads, seq, head_dim) in [(6, 64, 64), (8, 128, 64), (12, 256, 64)] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let v = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let id = format!("h{}s{}d{}", heads, seq, head_dim);

        group.throughput(Throughput::Elements((heads * seq * seq) as u64));

        // Standard attention
        group.bench_with_input(
            BenchmarkId::new("standard", &id),
            &(&q, &k, &v),
            |b, (q, k, v)| {
                b.iter(|| device.attention_ibp(black_box(q), black_box(k), black_box(v), scale))
            },
        );

        // Causal attention (decoder)
        group.bench_with_input(
            BenchmarkId::new("causal", &id),
            &(&q, &k, &v),
            |b, (q, k, v)| {
                b.iter(|| {
                    device.causal_attention_ibp(black_box(q), black_box(k), black_box(v), scale)
                })
            },
        );
    }
    group.finish();
}

fn bench_per_position_crown_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison/PerPositionCROWN");
    group.sample_size(10);

    let device = AcceleratedDevice::new();
    let hidden_dim = 384;
    let intermediate = 1536;

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

    for seq_len in [4, 16, 64] {
        let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

        group.throughput(Throughput::Elements(seq_len as u64));

        // Sequential (CPU baseline)
        group.bench_with_input(
            BenchmarkId::new("sequential", seq_len),
            &(&graph, &input),
            |b, (graph, input)| b.iter(|| graph.propagate_crown_per_position(black_box(input))),
        );

        // Parallel (Rayon)
        group.bench_with_input(
            BenchmarkId::new("parallel", seq_len),
            &(&graph, &input),
            |b, (graph, input)| {
                b.iter(|| device.crown_per_position_parallel(black_box(graph), black_box(input)))
            },
        );
    }
    group.finish();
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_scaling_seq_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/SeqLen");

    let device = AcceleratedDevice::new();
    let hidden_dim = 384;
    let intermediate = 1536;
    let weight = Array2::from_shape_fn((intermediate, hidden_dim), |_| 0.01_f32);
    let bias = Some(Array1::zeros(intermediate));

    for seq_len in [16, 64, 256, 1024, 4096] {
        let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("linear_ibp", seq_len),
            &input,
            |b, input| b.iter(|| device.linear_ibp(black_box(input), &weight, bias.as_ref())),
        );
    }
    group.finish();
}

fn bench_scaling_hidden_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/HiddenDim");

    let device = AcceleratedDevice::new();
    let seq_len = 128;

    for hidden_dim in [256, 384, 512, 768, 1024] {
        let intermediate = hidden_dim * 4;
        let weight = Array2::from_shape_fn((intermediate, hidden_dim), |_| 0.01_f32);
        let bias = Some(Array1::zeros(intermediate));
        let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

        group.throughput(Throughput::Elements(
            (seq_len * hidden_dim * intermediate) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("linear_ibp", hidden_dim),
            &(&input, &weight, &bias),
            |b, (input, weight, bias)| {
                b.iter(|| device.linear_ibp(black_box(input), black_box(weight), bias.as_ref()))
            },
        );
    }
    group.finish();
}

// ============================================================================
// GPU Kernel Microbenchmarks (CPU vs wgpu)
// ============================================================================

/// Isolated Linear IBP benchmark: CPU (Rayon) vs wgpu GPU
fn bench_gpu_linear_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("GpuKernels/Linear");

    let cpu_device = AcceleratedDevice::new();
    let gpu_device = match WgpuDevice::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("wgpu device not available: {e}");
            None
        }
    };

    // Test configurations: (in_features, out_features, batch*seq)
    for (in_dim, out_dim, batch_seq) in [
        (384, 1536, 64),  // Whisper-tiny MLP up-projection, seq=64
        (384, 1536, 256), // Whisper-tiny MLP up-projection, seq=256
        (1536, 384, 256), // Whisper-tiny MLP down-projection, seq=256
        (768, 3072, 128), // GPT-2 small MLP up-projection
        (3072, 768, 128), // GPT-2 small MLP down-projection
    ] {
        let weight = Array2::from_shape_fn((out_dim, in_dim), |(i, j)| {
            0.01 * ((i + j) % 10) as f32 - 0.05
        });
        let bias = Some(Array1::from_elem(out_dim, 0.01_f32));
        let input = make_input(&[1, batch_seq, in_dim], 0.5, 0.01);
        let id = format!("{in_dim}x{out_dim}_b{batch_seq}");

        group.throughput(Throughput::Elements((batch_seq * in_dim * out_dim) as u64));

        // CPU (Rayon parallel)
        group.bench_with_input(
            BenchmarkId::new("cpu", &id),
            &(&input, &weight, &bias),
            |b, (input, weight, bias)| {
                b.iter(|| cpu_device.linear_ibp(black_box(input), black_box(weight), bias.as_ref()))
            },
        );

        // wgpu GPU
        if let Some(ref gpu) = gpu_device {
            group.bench_with_input(
                BenchmarkId::new("wgpu", &id),
                &(&input, &weight, &bias),
                |b, (input, weight, bias)| {
                    b.iter(|| gpu.linear_ibp(black_box(input), black_box(weight), bias.as_ref()))
                },
            );
        }
    }
    group.finish();
}

/// Isolated MatMul IBP benchmark: CPU vs wgpu
fn bench_gpu_matmul_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("GpuKernels/MatMul");

    let cpu_device = AcceleratedDevice::new();
    let gpu_device = match WgpuDevice::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("wgpu device not available: {e}");
            None
        }
    };

    // Attention-style batched matmul: Q @ K^T
    // Shape: [batch, heads, seq, dim] @ [batch, heads, dim, seq] -> [batch, heads, seq, seq]
    for (heads, seq, head_dim) in [
        (6, 64, 64),   // Whisper-tiny encoder
        (6, 128, 64),  // Whisper-tiny encoder, longer
        (6, 256, 64),  // Whisper-tiny encoder, seq=256
        (8, 128, 64),  // Whisper-small
        (12, 128, 64), // GPT-2 small
    ] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k_t = make_input(&[1, heads, head_dim, seq], 0.5, 0.1);
        let id = format!("h{heads}s{seq}d{head_dim}");

        // FLOPs for batched matmul: 2 * batch * heads * seq * seq * head_dim
        group.throughput(Throughput::Elements(
            (2 * heads * seq * seq * head_dim) as u64,
        ));

        // CPU
        group.bench_with_input(BenchmarkId::new("cpu", &id), &(&q, &k_t), |b, (q, k_t)| {
            b.iter(|| cpu_device.matmul_ibp(black_box(q), black_box(k_t)))
        });

        // wgpu
        if let Some(ref gpu) = gpu_device {
            group.bench_with_input(BenchmarkId::new("wgpu", &id), &(&q, &k_t), |b, (q, k_t)| {
                b.iter(|| gpu.matmul_ibp(black_box(q), black_box(k_t)))
            });
        }
    }
    group.finish();
}

/// Isolated full Attention IBP benchmark: CPU vs wgpu
/// This includes transpose, matmul, scale, softmax, and final matmul.
fn bench_gpu_attention_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("GpuKernels/Attention");
    group.sample_size(20);

    let cpu_device = AcceleratedDevice::new();
    let gpu_device = match WgpuDevice::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("wgpu device not available: {e}");
            None
        }
    };

    // Full attention: softmax((Q @ K^T) * scale) @ V
    for (heads, seq, head_dim) in [
        (6, 64, 64),   // Whisper-tiny
        (6, 128, 64),  // Whisper-tiny, longer seq
        (6, 256, 64),  // Whisper-tiny, seq=256
        (8, 128, 64),  // Whisper-small
        (12, 128, 64), // GPT-2 small style
    ] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let v = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let id = format!("h{heads}s{seq}d{head_dim}");

        // Attention FLOPs: 4 * batch * heads * seq^2 * head_dim (two matmuls)
        group.throughput(Throughput::Elements(
            (4 * heads * seq * seq * head_dim) as u64,
        ));

        // CPU (full attention including softmax)
        group.bench_with_input(
            BenchmarkId::new("cpu", &id),
            &(&q, &k, &v),
            |b, (q, k, v)| {
                b.iter(|| cpu_device.attention_ibp(black_box(q), black_box(k), black_box(v), scale))
            },
        );

        // wgpu
        if let Some(ref gpu) = gpu_device {
            group.bench_with_input(
                BenchmarkId::new("wgpu", &id),
                &(&q, &k, &v),
                |b, (q, k, v)| {
                    b.iter(|| gpu.attention_ibp(black_box(q), black_box(k), black_box(v), scale))
                },
            );
        }
    }
    group.finish();
}

/// Causal attention benchmark: CPU vs wgpu
fn bench_gpu_causal_attention_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("GpuKernels/CausalAttention");
    group.sample_size(20);

    let cpu_device = AcceleratedDevice::new();
    let gpu_device = match WgpuDevice::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("wgpu device not available: {e}");
            None
        }
    };

    for (heads, seq, head_dim) in [(6, 64, 64), (6, 128, 64), (12, 128, 64)] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let v = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let id = format!("h{heads}s{seq}d{head_dim}");

        group.throughput(Throughput::Elements(
            (4 * heads * seq * seq * head_dim) as u64,
        ));

        // CPU
        group.bench_with_input(
            BenchmarkId::new("cpu", &id),
            &(&q, &k, &v),
            |b, (q, k, v)| {
                b.iter(|| {
                    cpu_device.causal_attention_ibp(black_box(q), black_box(k), black_box(v), scale)
                })
            },
        );

        // wgpu
        if let Some(ref gpu) = gpu_device {
            group.bench_with_input(
                BenchmarkId::new("wgpu", &id),
                &(&q, &k, &v),
                |b, (q, k, v)| {
                    b.iter(|| {
                        gpu.causal_attention_ibp(black_box(q), black_box(k), black_box(v), scale)
                    })
                },
            );
        }
    }
    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    comparison_benches,
    bench_linear_comparison,
    bench_matmul_comparison,
    bench_attention_comparison,
    bench_causal_attention,
    bench_per_position_crown_comparison,
);

criterion_group!(
    scaling_benches,
    bench_scaling_seq_len,
    bench_scaling_hidden_dim,
);

criterion_group!(
    gpu_kernel_benches,
    bench_gpu_linear_ibp,
    bench_gpu_matmul_ibp,
    bench_gpu_attention_ibp,
    bench_gpu_causal_attention_ibp,
);

criterion_main!(comparison_benches, scaling_benches, gpu_kernel_benches);
