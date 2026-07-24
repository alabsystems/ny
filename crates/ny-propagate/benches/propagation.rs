// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for ny bound propagation
//!
//! Run with: cargo bench -p ny-propagate
//! HTML reports: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::layers::{
    Conv1dLayer, Conv2dLayer, GELULayer, LayerNormLayer, LinearLayer, MatMulLayer, ReLULayer,
    SoftmaxLayer,
};
use ny_propagate::{BoundPropagation, GraphNetwork, GraphNode, Layer, Network};
use ny_tensor::BoundedTensor;

/// Create a BoundedTensor with specified shape
fn make_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon).unwrap()
}

// ============================================================================
// IBP Benchmarks
// ============================================================================

fn bench_linear_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/Linear");

    for (in_dim, out_dim) in [(384, 1536), (1536, 384), (768, 3072), (3072, 768)] {
        let input = make_input(&[1, 64, in_dim], 0.5, 0.01);
        let weight = Array2::from_shape_fn((out_dim, in_dim), |_| 0.01_f32);
        let bias = Some(Array1::zeros(out_dim));
        let layer = LinearLayer::new(weight, bias).unwrap();

        group.throughput(Throughput::Elements((in_dim * out_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", in_dim, out_dim)),
            &(&layer, &input),
            |b, (layer, input)| b.iter(|| layer.propagate_ibp(black_box(input))),
        );
    }
    group.finish();
}

fn bench_relu_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/ReLU");
    let layer = ReLULayer;

    for size in [384, 1536, 3072] {
        let input = make_input(&[1, 64, size], 0.0, 0.5);
        group.throughput(Throughput::Elements((64 * size) as u64));
        group.bench_with_input(BenchmarkId::new("forward", size), &input, |b, input| {
            b.iter(|| layer.propagate_ibp(black_box(input)))
        });
    }
    group.finish();
}

fn bench_gelu_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/GELU");
    let layer = GELULayer::default();

    for size in [384, 1536, 3072] {
        let input = make_input(&[1, 64, size], 0.0, 0.5);
        group.throughput(Throughput::Elements((64 * size) as u64));
        group.bench_with_input(BenchmarkId::new("forward", size), &input, |b, input| {
            b.iter(|| layer.propagate_ibp(black_box(input)))
        });
    }
    group.finish();
}

fn bench_layernorm_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/LayerNorm");

    for hidden_dim in [384, 768, 1024] {
        let input = make_input(&[1, 64, hidden_dim], 0.5, 0.01);
        let layer =
            LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5).unwrap();

        group.throughput(Throughput::Elements((64 * hidden_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", hidden_dim),
            &(&layer, &input),
            |b, (layer, input)| b.iter(|| layer.propagate_ibp(black_box(input))),
        );
    }
    group.finish();
}

fn bench_conv1d_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/Conv1d");

    // Whisper-style conv: (in_channels, out_channels, kernel_size)
    for (in_ch, out_ch, kernel) in [(80, 384, 3), (384, 384, 3), (512, 512, 3)] {
        let seq_len = 1500; // Whisper max
        let input = make_input(&[1, in_ch, seq_len], 0.5, 0.01);
        let weight = ArrayD::from_shape_fn(IxDyn(&[out_ch, in_ch, kernel]), |_| 0.01_f32);
        let bias = Some(Array1::zeros(out_ch));
        let layer = Conv1dLayer::new(weight, bias, 1, 1).unwrap();

        group.throughput(Throughput::Elements(
            (in_ch * out_ch * kernel * seq_len) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}", in_ch, out_ch, kernel)),
            &(&layer, &input),
            |b, (layer, input)| b.iter(|| layer.propagate_ibp(black_box(input))),
        );
    }
    group.finish();
}

fn bench_conv2d_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/Conv2d");
    group.sample_size(20); // Slower, reduce samples

    for (in_ch, out_ch, kernel) in [(3, 64, 7), (64, 128, 3), (128, 256, 3)] {
        let input = make_input(&[1, in_ch, 32, 32], 0.5, 0.01);
        let weight = ArrayD::from_shape_fn(IxDyn(&[out_ch, in_ch, kernel, kernel]), |_| 0.01_f32);
        let bias = Some(Array1::zeros(out_ch));
        let layer = Conv2dLayer::new(weight, bias, (1, 1), (kernel / 2, kernel / 2)).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}", in_ch, out_ch, kernel)),
            &(&layer, &input),
            |b, (layer, input)| b.iter(|| layer.propagate_ibp(black_box(input))),
        );
    }
    group.finish();
}

// ============================================================================
// Attention Benchmarks
// ============================================================================

fn bench_matmul_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/MatMul");

    let matmul = MatMulLayer::new(false, None);

    // Q @ K^T shapes: [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq]
    for (heads, seq, head_dim) in [(6, 64, 64), (8, 128, 64), (12, 256, 64)] {
        let q = make_input(&[1, heads, seq, head_dim], 0.5, 0.1);
        let k = make_input(&[1, heads, head_dim, seq], 0.5, 0.1);

        group.throughput(Throughput::Elements((heads * seq * seq * head_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("QK", format!("h{}s{}d{}", heads, seq, head_dim)),
            &(&q, &k),
            |b, (q, k)| b.iter(|| matmul.propagate_ibp_binary(black_box(q), black_box(k))),
        );
    }
    group.finish();
}

fn bench_softmax_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/Softmax");

    let softmax = SoftmaxLayer::new(-1);

    for (heads, seq) in [(6, 64), (8, 128), (12, 256)] {
        let input = make_input(&[1, heads, seq, seq], 0.0, 1.0);

        group.throughput(Throughput::Elements((heads * seq * seq) as u64));
        group.bench_with_input(
            BenchmarkId::new("attention_weights", format!("h{}s{}", heads, seq)),
            &input,
            |b, input| b.iter(|| softmax.propagate_ibp(black_box(input))),
        );
    }
    group.finish();
}

// ============================================================================
// CROWN Benchmarks
// ============================================================================

fn bench_mlp_crown(c: &mut Criterion) {
    let mut group = c.benchmark_group("CROWN/MLP");
    group.sample_size(20);

    for hidden_dim in [384, 768] {
        let intermediate = hidden_dim * 4;

        // Build MLP: Linear -> GELU -> Linear
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

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::GELU(gelu));
        network.add_layer(Layer::Linear(linear2));

        // 1D input for CROWN
        let input = make_input(&[hidden_dim], 0.5, 0.01);

        group.bench_with_input(
            BenchmarkId::new("forward", hidden_dim),
            &(&network, &input),
            |b, (network, input)| b.iter(|| network.propagate_crown(black_box(input))),
        );
    }
    group.finish();
}

fn bench_per_position_crown(c: &mut Criterion) {
    let mut group = c.benchmark_group("CROWN/PerPosition");
    group.sample_size(10); // Expensive

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
        group.bench_with_input(
            BenchmarkId::new("seq", seq_len),
            &(&graph, &input),
            |b, (graph, input)| b.iter(|| graph.propagate_crown_per_position(black_box(input))),
        );
    }
    group.finish();
}

// ============================================================================
// Full Pipeline Benchmarks
// ============================================================================

fn bench_encoder_block_ibp(c: &mut Criterion) {
    let mut group = c.benchmark_group("IBP/EncoderBlock");
    group.sample_size(20);

    // Whisper-tiny config
    let hidden_dim = 384;
    let intermediate = 1536;
    let _num_heads = 6;
    let _head_dim = 64;

    for seq_len in [64, 128, 256] {
        // Self-attention components
        let qkv_proj = LinearLayer::new(
            Array2::from_shape_fn((hidden_dim * 3, hidden_dim), |_| 0.01_f32),
            Some(Array1::zeros(hidden_dim * 3)),
        )
        .unwrap();
        let out_proj = LinearLayer::new(
            Array2::from_shape_fn((hidden_dim, hidden_dim), |_| 0.01_f32),
            Some(Array1::zeros(hidden_dim)),
        )
        .unwrap();

        // MLP
        let fc1 = LinearLayer::new(
            Array2::from_shape_fn((intermediate, hidden_dim), |_| 0.01_f32),
            Some(Array1::zeros(intermediate)),
        )
        .unwrap();
        let fc2 = LinearLayer::new(
            Array2::from_shape_fn((hidden_dim, intermediate), |_| 0.01_f32),
            Some(Array1::zeros(hidden_dim)),
        )
        .unwrap();

        let gelu = GELULayer::default();
        let ln1 =
            LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5).unwrap();
        let ln2 =
            LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5).unwrap();

        let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
        group.bench_with_input(BenchmarkId::new("full", seq_len), &input, |b, input| {
            b.iter(|| {
                // LayerNorm
                let normed = ln1.propagate_ibp(input).unwrap();
                // QKV projection
                let _qkv = qkv_proj.propagate_ibp(&normed).unwrap();
                // (skip attention for now - just projections)
                let out = out_proj.propagate_ibp(&normed).unwrap();
                // LayerNorm 2
                let normed2 = ln2.propagate_ibp(&out).unwrap();
                // MLP
                let h = fc1.propagate_ibp(&normed2).unwrap();
                let h = gelu.propagate_ibp(&h).unwrap();
                let _out = fc2.propagate_ibp(&h).unwrap();
            })
        });
    }
    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling");

    let hidden_dim = 384;
    let intermediate = 1536;

    let linear1 = LinearLayer::new(
        Array2::from_shape_fn((intermediate, hidden_dim), |_| 0.01_f32),
        Some(Array1::zeros(intermediate)),
    )
    .unwrap();

    // Test how IBP scales with batch/sequence dimensions
    for seq_len in [16, 64, 256, 1024] {
        let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("Linear_IBP", seq_len),
            &input,
            |b, input| b.iter(|| linear1.propagate_ibp(black_box(input))),
        );
    }
    group.finish();
}

// ============================================================================
// Memory Pool Benchmarks
// ============================================================================

fn bench_pool_allocation(c: &mut Criterion) {
    use ny_tensor::TensorPool;

    let mut group = c.benchmark_group("Pool");

    // Compare pooled vs standard allocation
    for size in [1000, 10000, 100000, 1000000] {
        // Standard Vec allocation
        group.bench_with_input(BenchmarkId::new("Vec_alloc", size), &size, |b, &size| {
            b.iter(|| {
                let v = vec![0.0f32; size];
                black_box(v)
            })
        });

        // Pooled allocation (acquire + drop)
        group.bench_with_input(
            BenchmarkId::new("Pool_acquire_drop", size),
            &size,
            |b, &size| {
                TensorPool::clear();
                b.iter(|| {
                    let buf = TensorPool::acquire(size);
                    black_box(buf)
                    // buf auto-returns on drop
                })
            },
        );

        // Pooled allocation with reuse (warm pool)
        group.bench_with_input(BenchmarkId::new("Pool_reuse", size), &size, |b, &size| {
            TensorPool::clear();
            // Warm up pool
            let warmup = TensorPool::acquire(size);
            drop(warmup);
            b.iter(|| {
                let buf = TensorPool::acquire(size);
                black_box(&buf);
                drop(buf);
            })
        });
    }
    group.finish();
}

fn bench_pool_vs_arrayd(c: &mut Criterion) {
    use ny_tensor::TensorPool;

    let mut group = c.benchmark_group("Pool_vs_ArrayD");

    // Compare standard zeros vs pooled zeros
    for shape in [(vec![100, 100], "100x100"), (vec![1000, 1000], "1000x1000")] {
        let (dims, label) = shape;

        // Standard ndarray zeros
        group.bench_with_input(
            BenchmarkId::new("ndarray_zeros", label),
            &dims,
            |b, dims| {
                b.iter(|| {
                    let arr = ArrayD::<f32>::zeros(IxDyn(dims));
                    black_box(arr)
                })
            },
        );

        // Pooled zeros
        group.bench_with_input(BenchmarkId::new("pooled_zeros", label), &dims, |b, dims| {
            b.iter(|| {
                let total = dims.iter().product::<usize>();
                let mut buf = TensorPool::acquire(total);
                buf.truncate(total);
                let arr = buf.into_arrayd(dims).unwrap();
                black_box(arr)
            })
        });
    }
    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    ibp_benches,
    bench_linear_ibp,
    bench_relu_ibp,
    bench_gelu_ibp,
    bench_layernorm_ibp,
    bench_conv1d_ibp,
    bench_conv2d_ibp,
);

criterion_group!(attention_benches, bench_matmul_ibp, bench_softmax_ibp,);

criterion_group!(crown_benches, bench_mlp_crown, bench_per_position_crown,);

criterion_group!(pipeline_benches, bench_encoder_block_ibp, bench_scaling,);

criterion_group!(pool_benches, bench_pool_allocation, bench_pool_vs_arrayd,);

// ============================================================================
// Parallel Position Benchmarks
// ============================================================================

fn bench_parallel_positions(c: &mut Criterion) {
    use ny_propagate::parallel::{ParallelConfig, ParallelVerifier};
    use ny_propagate::types::PropagationMethod;

    let mut group = c.benchmark_group("Parallel/Positions");
    group.sample_size(10); // Expensive

    let hidden_dim = 384;
    let intermediate = 1536;

    // Build MLP network
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

    for seq_len in [8, 32, 128] {
        let input = make_input(&[1, seq_len, hidden_dim], 0.5, 0.01);

        // Serial verification (force no parallelism)
        let serial_config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 99999, // Force serial
            ..Default::default()
        };
        let serial_verifier = ParallelVerifier::new(serial_config);

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("serial_IBP", seq_len),
            &(&graph, &input),
            |b, (graph, input)| {
                b.iter(|| {
                    serial_verifier.verify_positions_parallel(black_box(graph), black_box(input), 1)
                })
            },
        );

        // Parallel verification
        let parallel_config = ParallelConfig {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 1, // Force parallel
            ..Default::default()
        };
        let parallel_verifier = ParallelVerifier::new(parallel_config);

        group.bench_with_input(
            BenchmarkId::new("parallel_IBP", seq_len),
            &(&graph, &input),
            |b, (graph, input)| {
                b.iter(|| {
                    parallel_verifier.verify_positions_parallel(
                        black_box(graph),
                        black_box(input),
                        1,
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(parallel_benches, bench_parallel_positions,);

// ============================================================================
// Streaming Computation Benchmarks
// ============================================================================

fn bench_streaming_crown(c: &mut Criterion) {
    use ny_propagate::streaming::{estimate_memory_savings, StreamingConfig, StreamingVerifier};

    let mut group = c.benchmark_group("Streaming/CROWN");
    group.sample_size(10); // CROWN is expensive

    let hidden_dim = 384;
    let intermediate = 1536;

    // Build a deeper network to show streaming benefits
    for num_layers in [5, 10, 20] {
        let mut network = Network::new();

        for i in 0..num_layers {
            let (in_dim, out_dim) = if i == 0 {
                (hidden_dim, intermediate)
            } else if i == num_layers - 1 {
                (
                    if i % 2 == 1 { intermediate } else { hidden_dim },
                    hidden_dim,
                )
            } else if i % 2 == 1 {
                (intermediate, hidden_dim)
            } else {
                (hidden_dim, intermediate)
            };

            let linear = LinearLayer::new(
                Array2::from_shape_fn((out_dim, in_dim), |_| 0.01_f32),
                Some(Array1::zeros(out_dim)),
            )
            .unwrap();
            network.add_layer(Layer::Linear(linear));
        }

        let input = make_input(&[hidden_dim], 0.5, 0.01);

        // Regular CROWN (no streaming)
        group.bench_with_input(
            BenchmarkId::new("regular", num_layers),
            &(&network, &input),
            |b, (network, input)| b.iter(|| network.propagate_crown(black_box(input))),
        );

        // Streaming CROWN with different intervals
        for interval in [2, 5, 10] {
            if interval > num_layers {
                continue;
            }

            let config = StreamingConfig {
                checkpoint_interval: interval,
                ..Default::default()
            };
            let verifier = StreamingVerifier::new(config);

            group.bench_with_input(
                BenchmarkId::new(format!("streaming_interval{}", interval), num_layers),
                &(&network, &input),
                |b, (network, input)| {
                    b.iter(|| {
                        verifier.propagate_crown_streaming(black_box(network), black_box(input))
                    })
                },
            );
        }

        // Report estimated memory savings
        let tensor_elements = hidden_dim; // Rough estimate
        let (original, streaming, savings) =
            estimate_memory_savings(num_layers, tensor_elements, 5);
        eprintln!(
            "  {} layers: original {}KB, streaming {}KB, savings {:.1}%",
            num_layers,
            original / 1024,
            streaming / 1024,
            savings
        );
    }
    group.finish();
}

fn bench_streaming_memory_tradeoff(c: &mut Criterion) {
    use ny_propagate::streaming::{StreamingConfig, StreamingVerifier};

    let mut group = c.benchmark_group("Streaming/MemoryTradeoff");
    group.sample_size(10);

    let hidden_dim = 256;
    let num_layers = 50;

    // Build a 50-layer network
    let mut network = Network::new();
    for _i in 0..num_layers {
        let linear = LinearLayer::new(
            Array2::from_shape_fn((hidden_dim, hidden_dim), |_| 0.01_f32),
            Some(Array1::zeros(hidden_dim)),
        )
        .unwrap();
        network.add_layer(Layer::Linear(linear));
    }

    let input = make_input(&[hidden_dim], 0.5, 0.01);

    // Test different checkpoint intervals to show memory-compute tradeoff
    // Smaller interval = more memory, less recomputation
    // Larger interval = less memory, more recomputation
    for interval in [1, 2, 5, 10, 25, 50] {
        let config = StreamingConfig {
            checkpoint_interval: interval,
            ..Default::default()
        };
        let verifier = StreamingVerifier::new(config);

        group.bench_with_input(
            BenchmarkId::new("interval", interval),
            &(&network, &input),
            |b, (network, input)| {
                b.iter(|| verifier.propagate_crown_streaming(black_box(network), black_box(input)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    streaming_benches,
    bench_streaming_crown,
    bench_streaming_memory_tradeoff,
);

criterion_main!(
    ibp_benches,
    attention_benches,
    crown_benches,
    pipeline_benches,
    pool_benches,
    parallel_benches,
    streaming_benches
);
