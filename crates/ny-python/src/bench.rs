// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::repr::repr_string;
use ndarray::{Array1, Array2, IxDyn as NdIxDyn};
use ny_propagate::layers::{GELULayer, LayerNormLayer, LinearLayer, MatMulLayer, SoftmaxLayer};
use ny_propagate::{BoundPropagation, Layer, Network};
use ny_tensor::BoundedTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

/// Single benchmark result item.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct BenchResultItem {
    /// Name of the benchmark
    #[pyo3(get)]
    pub name: String,

    /// Number of iterations run
    #[pyo3(get)]
    pub iterations: usize,

    /// Time per iteration in nanoseconds
    #[pyo3(get)]
    pub per_iter_ns: u64,

    /// Time per iteration in microseconds
    #[pyo3(get)]
    pub per_iter_us: f64,

    /// Time per iteration in milliseconds
    #[pyo3(get)]
    pub per_iter_ms: f64,

    /// Total time in nanoseconds
    #[pyo3(get)]
    pub total_ns: u64,

    /// Total time in milliseconds
    #[pyo3(get)]
    pub total_ms: f64,
}

#[pymethods]
impl BenchResultItem {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "BenchResultItem(name={}, per_iter_ms={:.3}, iterations={})",
            repr_string(&self.name),
            self.per_iter_ms,
            self.iterations
        )
    }
}

/// Dimensions used for benchmarks.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct BenchDimensions {
    /// Batch size
    #[pyo3(get)]
    pub batch: usize,

    /// Sequence length
    #[pyo3(get)]
    pub seq_len: usize,

    /// Hidden dimension
    #[pyo3(get)]
    pub hidden_dim: usize,

    /// Intermediate (feedforward) dimension
    #[pyo3(get)]
    pub intermediate_dim: usize,

    /// Number of attention heads
    #[pyo3(get)]
    pub num_heads: usize,

    /// Dimension per head
    #[pyo3(get)]
    pub head_dim: usize,

    /// Epsilon perturbation used
    #[pyo3(get)]
    pub epsilon: f32,
}

#[pymethods]
impl BenchDimensions {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "BenchDimensions(batch={}, seq={}, hidden={}, intermediate={}, heads={}, head_dim={}, eps={:.2e})",
            self.batch,
            self.seq_len,
            self.hidden_dim,
            self.intermediate_dim,
            self.num_heads,
            self.head_dim,
            self.epsilon
        )
    }
}

/// Full benchmark result.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct BenchResult {
    /// Type of benchmark (layer, attention, full)
    #[pyo3(get)]
    pub benchmark_type: String,

    /// Dimensions used for the benchmark
    #[pyo3(get)]
    pub dimensions: BenchDimensions,

    /// Individual benchmark results
    #[pyo3(get)]
    pub results: Vec<BenchResultItem>,
}

#[pymethods]
impl BenchResult {
    pub(crate) fn __repr__(&self) -> String {
        format!(
            "BenchResult(type={}, results={})",
            repr_string(&self.benchmark_type),
            self.results.len()
        )
    }

    /// Get a summary of all benchmark results
    pub(crate) fn summary(&self) -> String {
        let mut lines = vec![
            format!("Benchmark: {}", self.benchmark_type),
            format!("Dimensions: {}", self.dimensions.__repr__()),
            "Results:".to_string(),
        ];
        for r in &self.results {
            lines.push(format!(
                "  {}: {:.3}ms/iter ({} iters)",
                r.name, r.per_iter_ms, r.iterations
            ));
        }
        lines.join("\n")
    }
}

/// Helper to create BoundedTensor for benchmarks
fn make_bench_input(shape: &[usize], center: f32, epsilon: f32) -> ny_core::Result<BoundedTensor> {
    let values = ndarray::ArrayD::from_elem(NdIxDyn(shape), center);
    BoundedTensor::from_epsilon(values, epsilon)
}

/// Helper to run a benchmark with warmup
fn run_bench<F>(name: &str, iterations: usize, mut f: F) -> ny_core::Result<BenchResultItem>
where
    F: FnMut() -> ny_core::Result<()>,
{
    if iterations == 0 {
        return Err(ny_core::NyError::InvalidSpec(
            "benchmark iterations must be > 0".to_string(),
        ));
    }
    // Warmup
    for _ in 0..3 {
        f()?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f()?;
    }
    let elapsed = start.elapsed();
    let per_iter_ns = (elapsed.as_nanos() / iterations as u128) as u64;
    let total_ns = elapsed.as_nanos() as u64;

    Ok(BenchResultItem {
        name: name.to_string(),
        iterations,
        per_iter_ns,
        per_iter_us: per_iter_ns as f64 / 1000.0,
        per_iter_ms: per_iter_ns as f64 / 1_000_000.0,
        total_ns,
        total_ms: total_ns as f64 / 1_000_000.0,
    })
}

/// Run ny benchmarks.
///
/// Runs performance benchmarks for neural network verification operations.
///
/// Args:
///     benchmark_type: Type of benchmark to run. Options:
///         - "layer" (default): Individual layer IBP performance
///         - "attention": Attention component (MatMul, Softmax) performance
///         - "full": Full pipeline scaling tests
///
/// Returns:
///     BenchResult with timing information for each benchmark
///
/// Example:
///     >>> result = ny.bench()  # Run layer benchmarks
///     >>> print(result.summary())
///     >>> for r in result.results:
///     ...     print(f"{r.name}: {r.per_iter_ms:.3f}ms")
///
///     >>> result = ny.bench("attention")  # Run attention benchmarks
///     >>> result = ny.bench("full")  # Run full pipeline benchmarks
#[pyfunction]
#[pyo3(name = "bench", signature = (benchmark_type="layer"))]
pub fn run_benchmark(py: Python<'_>, benchmark_type: &str) -> PyResult<BenchResult> {
    // Whisper-tiny dimensions
    let batch = 1;
    let seq_len = 16;
    let hidden_dim = 384;
    let intermediate_dim = 1536;
    let num_heads = 6;
    let head_dim = 64;
    let epsilon = 0.01_f32;

    let dimensions = BenchDimensions {
        batch,
        seq_len,
        hidden_dim,
        intermediate_dim,
        num_heads,
        head_dim,
        epsilon,
    };

    let result = Python::detach(py, || -> ny_core::Result<BenchResult> {
        let mut results: Vec<BenchResultItem> = Vec::new();

        // Create common layers
        let linear_weight = Array2::from_shape_fn((intermediate_dim, hidden_dim), |_| 0.01_f32);
        let linear_bias = Some(Array1::zeros(intermediate_dim));
        let linear1 = LinearLayer::new(linear_weight, linear_bias)?;

        let linear_weight2 = Array2::from_shape_fn((hidden_dim, intermediate_dim), |_| 0.01_f32);
        let linear_bias2 = Some(Array1::zeros(hidden_dim));
        let linear2 = LinearLayer::new(linear_weight2, linear_bias2)?;

        let gelu = GELULayer::default();
        let layernorm =
            LayerNormLayer::new(Array1::ones(hidden_dim), Array1::zeros(hidden_dim), 1e-5).unwrap();

        match benchmark_type {
            "layer" => {
                let input = make_bench_input(&[batch, seq_len, hidden_dim], 0.5, epsilon)?;

                // Linear layer
                let mut linear_output = input.clone();
                results.push(run_bench("Linear IBP [384->1536]", 100, || {
                    linear_output = linear1.propagate_ibp(&input)?;
                    Ok(())
                })?);

                // GELU
                results.push(run_bench("GELU IBP [1536]", 100, || {
                    let _ = gelu.propagate_ibp(&linear_output)?;
                    Ok(())
                })?);
                let gelu_output = gelu.propagate_ibp(&linear_output)?;

                // Linear back
                results.push(run_bench("Linear IBP [1536->384]", 100, || {
                    let _ = linear2.propagate_ibp(&gelu_output)?;
                    Ok(())
                })?);
                let final_output = linear2.propagate_ibp(&gelu_output)?;

                // LayerNorm
                results.push(run_bench("LayerNorm IBP [384]", 100, || {
                    let _ = layernorm.propagate_ibp(&final_output)?;
                    Ok(())
                })?);

                // Full MLP
                let mut mlp = Network::new();
                mlp.add_layer(Layer::Linear(linear1));
                mlp.add_layer(Layer::GELU(gelu));
                mlp.add_layer(Layer::Linear(linear2));

                results.push(run_bench("Full MLP IBP [384->1536->384]", 100, || {
                    let _ = mlp.propagate_ibp(&input)?;
                    Ok(())
                })?);
            }

            "attention" => {
                // MatMul: Q @ K^T
                let q_input = make_bench_input(&[batch, num_heads, seq_len, head_dim], 0.5, 0.1)?;
                let k_input = make_bench_input(&[batch, num_heads, head_dim, seq_len], 0.5, 0.1)?;

                let matmul = MatMulLayer::new(false, None);

                results.push(run_bench(
                    &format!(
                        "MatMul IBP [{},{},{},{}] @ [{},{},{},{}]",
                        batch, num_heads, seq_len, head_dim, batch, num_heads, head_dim, seq_len
                    ),
                    100,
                    || {
                        let _ = matmul.propagate_ibp_binary(&q_input, &k_input)?;
                        Ok(())
                    },
                )?);

                // Softmax
                let attn_input = make_bench_input(&[batch, num_heads, seq_len, seq_len], 0.0, 1.0)?;
                let softmax = SoftmaxLayer::new(-1);

                results.push(run_bench(
                    &format!(
                        "Softmax IBP [{},{},{},{}]",
                        batch, num_heads, seq_len, seq_len
                    ),
                    100,
                    || {
                        let _ = softmax.propagate_ibp(&attn_input)?;
                        Ok(())
                    },
                )?);

                // MatMul scaling
                for seq in [4, 16, 64] {
                    let q = make_bench_input(&[batch, num_heads, seq, head_dim], 0.5, 0.1)?;
                    let k = make_bench_input(&[batch, num_heads, head_dim, seq], 0.5, 0.1)?;
                    let iterations = if seq <= 16 { 100 } else { 20 };

                    results.push(run_bench(
                        &format!("MatMul IBP seq={}", seq),
                        iterations,
                        || {
                            let _ = matmul.propagate_ibp_binary(&q, &k)?;
                            Ok(())
                        },
                    )?);
                }
            }

            "full" => {
                let mut mlp = Network::new();
                mlp.add_layer(Layer::Linear(linear1));
                mlp.add_layer(Layer::GELU(gelu));
                mlp.add_layer(Layer::Linear(linear2));

                // IBP scaling
                for seq in [4, 16, 64, 128] {
                    let input = make_bench_input(&[batch, seq, hidden_dim], 0.5, epsilon)?;
                    let iterations = if seq <= 16 {
                        100
                    } else if seq <= 64 {
                        20
                    } else {
                        5
                    };

                    results.push(run_bench(
                        &format!("MLP IBP seq={}", seq),
                        iterations,
                        || {
                            let _ = mlp.propagate_ibp(&input)?;
                            Ok(())
                        },
                    )?);
                }

                // CROWN 1-D
                let input_1d = make_bench_input(&[hidden_dim], 0.5, epsilon)?;
                results.push(run_bench("Full MLP CROWN 1-D [384]", 100, || {
                    let _ = mlp.propagate_crown(&input_1d)?;
                    Ok(())
                })?);
            }

            other => {
                return Err(ny_core::NyError::InvalidConfig(format!(
                    "Unknown benchmark type: '{other}'. Valid types: 'layer', 'attention', 'full'"
                )))
            }
        }

        Ok(BenchResult {
            benchmark_type: benchmark_type.to_string(),
            dimensions,
            results,
        })
    })
    .map_err(|e| PyValueError::new_err(format!("Benchmark error: {}", e)))?;

    Ok(result)
}
