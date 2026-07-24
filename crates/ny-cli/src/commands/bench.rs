// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Benchmark commands for measuring bound propagation performance.
//!
//! Provides CLI handlers for:
//! - `bench layer` - Individual layer IBP performance
//! - `bench attention` - Attention component (MatMul, Softmax) performance
//! - `bench full` - Full pipeline scaling tests

use super::backend::{resolve_gemm_backend, GemmBackendResolution};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_gpu::ComputeDevice;
use ny_propagate::layers::{GELULayer, LayerNormLayer, LinearLayer, MatMulLayer, SoftmaxLayer};
use ny_propagate::{BoundPropagation, Layer, Network};
use ny_tensor::BoundedTensor;
use std::time::Instant;

#[cfg(test)]
use super::backend::resolve_gemm_backend_with_factory;
use crate::BackendArg;

/// Benchmark result for JSON output
pub(crate) struct BenchResult {
    pub(crate) name: String,
    pub(crate) iterations: usize,
    pub(crate) per_iter_ns: u64,
    pub(crate) total_ns: u64,
}

/// Run a benchmark function with warmup and timing
pub(crate) fn bench_collect<F: FnMut() -> Result<()>>(
    name: &str,
    iterations: usize,
    mut f: F,
) -> Result<BenchResult> {
    anyhow::ensure!(
        iterations > 0,
        "benchmark '{name}' requires at least one iteration"
    );
    // Warmup
    for _ in 0..3 {
        f().with_context(|| format!("benchmark '{name}' warmup failed"))?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f().with_context(|| format!("benchmark '{name}' iteration failed"))?;
    }
    let elapsed = start.elapsed();
    // SAFETY(as u32): saturate to u32::MAX. Benchmarks never run 4B iterations
    // in practice, but this prevents silent truncation on pathological input.
    let per_iter = elapsed / iterations.min(u32::MAX as usize) as u32;

    Ok(BenchResult {
        name: name.to_string(),
        iterations,
        per_iter_ns: per_iter.as_nanos() as u64,
        total_ns: elapsed.as_nanos() as u64,
    })
}

/// Run a benchmark and optionally print results
pub(crate) fn bench<F: FnMut() -> Result<()>>(
    name: &str,
    iterations: usize,
    f: F,
    json: bool,
    results: &mut Vec<BenchResult>,
) -> Result<()> {
    let result = bench_collect(name, iterations, f)?;
    if !json {
        println!(
            "{}: {:?} per iteration ({} iterations)",
            name,
            std::time::Duration::from_nanos(result.per_iter_ns),
            iterations
        );
    }
    results.push(result);
    Ok(())
}

/// Create a BoundedTensor with specified shape from center and epsilon
pub(crate) fn make_bench_input(
    shape: &[usize],
    center: f32,
    epsilon: f32,
) -> Result<BoundedTensor> {
    let values = ArrayD::from_elem(IxDyn(shape), center);
    Ok(BoundedTensor::from_epsilon(values, epsilon)?)
}

/// Run benchmarks based on the benchmark type
// Justification: Benchmark runner accepts all VNN-COMP configuration parameters
// (benchmark suite, year, timeout, filters, branching strategy, attack config, etc.).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_benchmarks(
    benchmark: &str,
    json: bool,
    year: u32,
    timeout: Option<u64>,
    include_results: bool,
    model_filter: Option<&str>,
    property_filter: Option<&str>,
    branching: &str,
    max_domains: usize,
    proactive_cuts: bool,
    max_proactive_cuts: Option<usize>,
    relaxed_clip: bool,
    pgd_attack: bool,
    pgd_restarts: Option<usize>,
    gpu_bab: bool,
    no_la_warm_start: bool,
    backend: BackendArg,
    gpu: bool,
) -> Result<()> {
    if !json {
        println!("ny Benchmark Suite");
        println!("========================\n");
    }

    // Whisper-tiny dimensions
    let batch = 1;
    let seq_len = 16;
    let hidden_dim = 384;
    let intermediate_dim = 1536;
    let num_heads = 6;
    let head_dim = 64;
    let epsilon = 0.01_f32;

    if !json {
        println!("Dimensions:");
        println!(
            "  batch={}, seq={}, hidden={}, intermediate={}",
            batch, seq_len, hidden_dim, intermediate_dim
        );
        println!("  heads={}, head_dim={}\n", num_heads, head_dim);
    }

    // Create common layers
    let linear_weight = Array2::from_shape_fn((intermediate_dim, hidden_dim), |_| 0.01_f32);
    let linear_bias = Some(Array1::zeros(intermediate_dim));
    let linear1 = LinearLayer::new(linear_weight, linear_bias)?;

    let linear_weight2 = Array2::from_shape_fn((hidden_dim, intermediate_dim), |_| 0.01_f32);
    let linear_bias2 = Some(Array1::zeros(hidden_dim));
    let linear2 = LinearLayer::new(linear_weight2, linear_bias2)?;

    let gelu = GELULayer::default();
    let layernorm = LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5)?;

    // Collect all benchmark results
    let mut results: Vec<BenchResult> = Vec::new();
    let mut unknown_type = false;

    // `--backend`/`--gpu` selects the GEMM engine for the CROWN microbenchmark
    // in the `full` suite only. The IBP-only `layer`/`attention` suites and the
    // acasxu harness (whose GPU opt-in is `--gpu-bab`) never consume it, so a
    // non-default request there must be called out rather than silently run on
    // the CPU (a quiet drop would fabricate CPU-vs-wgpu A/B comparisons).
    let non_default_backend = backend != BackendArg::Cpu || gpu;

    match benchmark {
        "layer" => {
            if non_default_backend {
                eprintln!(
                    "Warning: --backend/--gpu is ignored by --benchmark layer (IBP-only, CPU)."
                );
            }
            if !json {
                println!("=== Layer Benchmarks (IBP) ===\n");
            }

            let input = make_bench_input(&[batch, seq_len, hidden_dim], 0.5, epsilon)?;

            // Linear layer
            let mut linear_output = input.clone();
            bench(
                "Linear IBP [384->1536]",
                100,
                || {
                    linear_output = linear1.propagate_ibp(&input)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;

            // GELU
            bench(
                "GELU IBP [1536]",
                100,
                || {
                    gelu.propagate_ibp(&linear_output)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;
            let gelu_output = gelu.propagate_ibp(&linear_output)?;

            // Linear back
            bench(
                "Linear IBP [1536->384]",
                100,
                || {
                    linear2.propagate_ibp(&gelu_output)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;
            let final_output = linear2.propagate_ibp(&gelu_output)?;

            // LayerNorm
            bench(
                "LayerNorm IBP [384]",
                100,
                || {
                    layernorm.propagate_ibp(&final_output)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;

            if !json {
                println!("\n=== Full MLP Path IBP ===\n");
            }

            let mut mlp = Network::new();
            mlp.add_layer(Layer::Linear(linear1.clone()));
            mlp.add_layer(Layer::GELU(gelu.clone()));
            mlp.add_layer(Layer::Linear(linear2.clone()));

            bench(
                "Full MLP IBP [384->1536->384]",
                100,
                || {
                    mlp.propagate_ibp(&input)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;
        }

        "attention" => {
            if non_default_backend {
                eprintln!(
                    "Warning: --backend/--gpu is ignored by --benchmark attention (IBP-only, CPU)."
                );
            }
            if !json {
                println!("=== Attention Component Benchmarks (IBP) ===\n");
            }

            // MatMul: Q @ K^T
            let q_input = make_bench_input(&[batch, num_heads, seq_len, head_dim], 0.5, 0.1)?;
            let k_input = make_bench_input(&[batch, num_heads, head_dim, seq_len], 0.5, 0.1)?;

            let matmul = MatMulLayer::new(false, None);

            bench(
                &format!(
                    "MatMul IBP [{},{},{},{}] @ [{},{},{},{}]",
                    batch, num_heads, seq_len, head_dim, batch, num_heads, head_dim, seq_len
                ),
                100,
                || {
                    matmul.propagate_ibp_binary(&q_input, &k_input)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;

            // Softmax
            let attn_input = make_bench_input(&[batch, num_heads, seq_len, seq_len], 0.0, 1.0)?;
            let softmax = SoftmaxLayer::new(-1);

            bench(
                &format!(
                    "Softmax IBP [{},{},{},{}]",
                    batch, num_heads, seq_len, seq_len
                ),
                100,
                || {
                    softmax.propagate_ibp(&attn_input)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;

            if !json {
                println!("\n=== MatMul Scaling ===\n");
            }

            for seq in [4, 16, 64] {
                let q = make_bench_input(&[batch, num_heads, seq, head_dim], 0.5, 0.1)?;
                let k = make_bench_input(&[batch, num_heads, head_dim, seq], 0.5, 0.1)?;
                let iterations = if seq <= 16 { 100 } else { 20 };

                bench(
                    &format!("MatMul IBP seq={}", seq),
                    iterations,
                    || {
                        matmul.propagate_ibp_binary(&q, &k)?;
                        Ok(())
                    },
                    json,
                    &mut results,
                )?;
            }
        }

        "full" => {
            if !json {
                println!("=== Full Pipeline Benchmarks ===\n");
            }

            let mut mlp = Network::new();
            mlp.add_layer(Layer::Linear(linear1.clone()));
            mlp.add_layer(Layer::GELU(gelu.clone()));
            mlp.add_layer(Layer::Linear(linear2.clone()));

            if !json {
                println!("=== IBP Scaling ===\n");
            }

            for seq in [4, 16, 64, 128] {
                let input = make_bench_input(&[batch, seq, hidden_dim], 0.5, epsilon)?;
                let iterations = if seq <= 16 {
                    100
                } else if seq <= 64 {
                    20
                } else {
                    5
                };

                bench(
                    &format!("MLP IBP seq={}", seq),
                    iterations,
                    || {
                        mlp.propagate_ibp(&input)?;
                        Ok(())
                    },
                    json,
                    &mut results,
                )?;
            }

            if !json {
                println!("\n=== CROWN (1-D) ===\n");
            }

            let crown_backend = resolve_crown_benchmark_backend(backend, gpu, json);
            let gemm_engine = crown_backend.gemm_engine();
            let input_1d = make_bench_input(&[hidden_dim], 0.5, epsilon)?;
            bench(
                "Full MLP CROWN 1-D [384]",
                100,
                || {
                    mlp.propagate_crown_with_engine(&input_1d, gemm_engine)?;
                    Ok(())
                },
                json,
                &mut results,
            )?;
        }

        "acasxu" => {
            // Delegate to bench_acasxu module
            use super::bench_acasxu::{print_summary, run_acasxu_benchmark, AcasxuBenchmarkArgs};

            if non_default_backend {
                eprintln!(
                    "Warning: --backend/--gpu is ignored by --benchmark acasxu; \
                     use --gpu-bab for the wgpu BaB engine."
                );
            }

            let args = AcasxuBenchmarkArgs {
                timeout_override: timeout,
                year,
                json,
                include_results,
                branching: branching.to_string(),
                model_filter: model_filter.map(|s| s.to_string()),
                property_filter: property_filter.map(|s| s.to_string()),
                max_domains,
                proactive_cuts,
                max_proactive_cuts: max_proactive_cuts.unwrap_or(100),
                relaxed_clip,
                pgd_attack,
                pgd_restarts: pgd_restarts.unwrap_or(100),
                gpu_bab,
                no_la_warm_start,
                ..Default::default()
            };

            match run_acasxu_benchmark(args) {
                Ok(summary) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&summary)?);
                    } else {
                        print_summary(&summary);
                    }
                }
                Err(e) => {
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark_type": "acasxu",
                                "error": e.to_string()
                            })
                        );
                    } else {
                        eprintln!("ACAS-Xu benchmark failed: {}", e);
                    }
                    anyhow::bail!("ACAS-Xu benchmark failed: {}", e);
                }
            }
        }

        _ => {
            unknown_type = true;
            if !json {
                println!(
                    "Unknown benchmark type: {}. Available: layer, attention, full, acasxu",
                    benchmark
                );
            } else {
                println!(
                    "{}",
                    serde_json::json!({
                        "benchmark_type": benchmark,
                        "valid_type": false,
                        "error": format!(
                            "Unknown benchmark type: {}. Available: layer, attention, full, acasxu",
                            benchmark
                        )
                    })
                );
            }
        }
    }

    if json {
        let results_json: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "name": r.name,
                    "iterations": r.iterations,
                    "per_iter_ns": r.per_iter_ns,
                    "per_iter_us": r.per_iter_ns as f64 / 1000.0,
                    "per_iter_ms": r.per_iter_ns as f64 / 1_000_000.0,
                    "total_ns": r.total_ns,
                    "total_ms": r.total_ns as f64 / 1_000_000.0
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::json!({
                "benchmark_type": benchmark,
                "valid_type": !unknown_type,
                "dimensions": {
                    "batch": batch,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "intermediate_dim": intermediate_dim,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "epsilon": epsilon
                },
                "results": results_json
            })
        );
    } else {
        println!("\n=== Summary ===");
        println!("Benchmark complete. Use --benchmark <type> to run specific benchmarks.");
        println!("  layer     - Individual layer IBP performance");
        println!("  attention - Attention component (MatMul, Softmax) performance");
        println!("  full      - Full pipeline scaling tests");
    }

    if unknown_type {
        anyhow::bail!(
            "Unknown benchmark type: {}. Available: layer, attention, full, acasxu",
            benchmark
        );
    }

    Ok(())
}

fn resolve_crown_benchmark_backend(
    backend: BackendArg,
    gpu: bool,
    json: bool,
) -> GemmBackendResolution<ComputeDevice> {
    resolve_gemm_backend(backend, gpu, json)
}

#[cfg(test)]
pub(super) fn resolve_crown_benchmark_backend_with_factory<T, F>(
    backend: BackendArg,
    gpu: bool,
    json: bool,
    build_device: F,
) -> GemmBackendResolution<T>
where
    F: FnOnce(BackendArg) -> Result<T>,
{
    resolve_gemm_backend_with_factory(backend, gpu, json, build_device)
}
