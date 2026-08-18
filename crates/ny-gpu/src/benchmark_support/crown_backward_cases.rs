// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::benchmark_support::crown_backward_workloads::{
    bench_rng_f32, conv_output_dim, shape_product3, ConvBenchSpec, METAROOM_CASE_NAME,
    METAROOM_CONV_SPECS, METAROOM_HIDDEN_DIM, METAROOM_INPUT_SHAPE, METAROOM_OUTPUT_DIM,
    SOUNDNESSBENCH_CASE_NAME, SOUNDNESSBENCH_CONV_SPECS, SOUNDNESSBENCH_INPUT_DIM,
    SOUNDNESSBENCH_OUTPUT_DIM, SOUNDNESSBENCH_RESHAPE_SHAPE,
};
use crate::ComputeDevice;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_propagate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer, ReshapeLayer};
use ny_propagate::{GraphNetwork, Layer, Network};
use ny_tensor::BoundedTensor;

const CPU_CROWN_DENSE_BUDGET_ENV: &str = "NY_DENSE_BUDGET_MB";
const DEFAULT_CPU_CROWN_DENSE_BUDGET_MB: usize = 2048;

#[doc(hidden)]
pub struct BenchCase {
    name: &'static str,
    network: Network,
    input: BoundedTensor,
    parameter_count: usize,
    /// Conservative estimate of CPU CROWN backward peak memory in bytes (#3515).
    estimated_cpu_peak_bytes: usize,
}

impl BenchCase {
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    #[must_use]
    pub fn estimated_cpu_peak_bytes(&self) -> usize {
        self.estimated_cpu_peak_bytes
    }

    pub fn run_cpu_production(&self) -> Result<()> {
        self.network.propagate_crown(&self.input).map(|_| ())
    }

    pub fn run_gpu_production(&self, engine: &dyn GemmEngine) -> Result<()> {
        self.network
            .propagate_crown_with_engine(&self.input, Some(engine))
            .map(|_| ())
    }

    pub fn collect_ibp(&self) -> Result<()> {
        self.network.collect_ibp_bounds(&self.input).map(|_| ())
    }

    pub fn run_crown_ibp_from_fresh_ibp(&self, engine: &dyn GemmEngine) -> Result<()> {
        let ibp_bounds = self.network.collect_ibp_bounds(&self.input)?;
        self.network
            .collect_crown_ibp_bounds_with_precomputed_ibp(
                &self.input,
                ibp_bounds,
                Some(engine),
                None,
            )
            .map(|_| ())
    }

    pub fn run_production_from_fresh_ibp(&self, engine: &dyn GemmEngine) -> Result<()> {
        let ibp_bounds = self.network.collect_ibp_bounds(&self.input)?;
        self.network
            .propagate_crown_with_precomputed_ibp(&self.input, ibp_bounds, Some(engine), None)
            .map(|_| ())
    }

    /// Convert the sequential `Network` to a `GraphNetwork` for graph-path
    /// benchmarking (#3716). Real VNN-COMP ONNX models load as `GraphNetwork`,
    /// so this conversion lets us measure the actual code path that competition
    /// workloads exercise.
    pub fn to_graph_network(&self) -> Result<GraphNetwork> {
        GraphNetwork::from_sequential(&self.network)
    }

    /// Run IBP forward on the graph representation.
    pub fn run_graph_ibp(&self) -> Result<()> {
        let graph = self.to_graph_network()?;
        graph.propagate_ibp(&self.input).map(|_| ())
    }

    /// Run the O(N²) CROWN-IBP per-node tightening loop on the graph path.
    ///
    /// This is the critical path for real VNN-COMP models: for each intermediate
    /// node, it runs backward CROWN to the input then intersects with IBP bounds.
    /// The `engine` parameter threads GPU GEMM acceleration through per-node
    /// backward passes (#3549).
    pub fn run_graph_crown_ibp_collection(&self, engine: Option<&dyn GemmEngine>) -> Result<()> {
        let graph = self.to_graph_network()?;
        graph
            .collect_crown_ibp_bounds_dag_with_engine(&self.input, engine)
            .map(|_| ())
    }

    /// Run full CROWN backward on the graph representation (alpha-CROWN → fallback).
    ///
    /// Measures the complete graph CROWN path that VNN-COMP alpha-CROWN uses:
    /// 1. Collect CROWN-IBP intermediate bounds (O(N²))
    /// 2. Run backward CROWN from output to input using tightened intermediates
    pub fn run_graph_crown(&self, engine: Option<&dyn GemmEngine>) -> Result<()> {
        let graph = self.to_graph_network()?;
        graph
            .propagate_crown_with_engine(&self.input, engine)
            .map(|_| ())
    }

    // Its only caller, `crown_backward_cases_tests`, is selected by the same
    // real-adapter conformance feature.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn assert_graph_matches_sequential(
        &self,
        engine: Option<&dyn GemmEngine>,
        epsilon: f32,
    ) -> Result<()> {
        let sequential_bounds = self.network.propagate_crown(&self.input)?;
        let graph = self.to_graph_network()?;
        let graph_bounds = graph.propagate_crown_with_engine(&self.input, engine)?;
        ensure_bounds_close(&graph_bounds, &sequential_bounds, epsilon, self.name)
    }

    pub fn assert_gpu_matches_cpu(
        &self,
        gpu_device: &ComputeDevice,
        engine: &dyn GemmEngine,
        epsilon: f32,
    ) -> Result<()> {
        let cpu_output = self.network.propagate_crown(&self.input)?;
        clear_gpu_crown_working_set(gpu_device)?;
        let gpu_output = self
            .network
            .propagate_crown_with_engine(&self.input, Some(engine))?;
        ensure_bounds_close(&gpu_output, &cpu_output, epsilon, self.name)
    }
}

// Mirrors ny-propagate's sequential CROWN dense-budget policy until the
// helper is exported through the committed public API.
#[doc(hidden)]
pub fn cpu_crown_dense_budget_bytes() -> usize {
    if let Ok(mb) = std::env::var(CPU_CROWN_DENSE_BUDGET_ENV) {
        if let Ok(parsed) = mb.parse::<usize>() {
            return parsed * 1024 * 1024;
        }
    }
    DEFAULT_CPU_CROWN_DENSE_BUDGET_MB * 1024 * 1024
}

/// Conservative CPU CROWN peak A-matrix estimate: `4 * dim² * sizeof(f32)`.
/// CROWN-IBP Dense coefficients are O(dim²); 4× for lower/upper × old/new.
fn estimate_dense_peak_bytes(max_dim: usize) -> usize {
    4 * max_dim
        .saturating_mul(max_dim)
        .saturating_mul(size_of::<f32>())
}

fn conv_max_output_dim(conv_specs: &[ConvBenchSpec]) -> usize {
    conv_specs
        .iter()
        .copied()
        .map(conv_output_dim)
        .max()
        .unwrap_or(0)
}

fn ensure_bounds_close(
    actual: &BoundedTensor,
    expected: &BoundedTensor,
    epsilon: f32,
    label: &str,
) -> Result<()> {
    if actual.shape() != expected.shape() {
        return Err(NyError::InternalError(format!(
            "{label}: shape mismatch actual={:?} expected={:?}",
            actual.shape(),
            expected.shape()
        )));
    }

    for (idx, (&actual_l, &expected_l)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        let diff = (actual_l - expected_l).abs();
        if diff > epsilon {
            return Err(NyError::InternalError(format!(
                "{label}: lower[{idx}] actual={actual_l} expected={expected_l} diff={diff} epsilon={epsilon}"
            )));
        }
    }

    for (idx, (&actual_u, &expected_u)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        let diff = (actual_u - expected_u).abs();
        if diff > epsilon {
            return Err(NyError::InternalError(format!(
                "{label}: upper[{idx}] actual={actual_u} expected={expected_u} diff={diff} epsilon={epsilon}"
            )));
        }
    }

    Ok(())
}

fn build_random_linear_layer(
    seed: &mut u64,
    out_dim: usize,
    in_dim: usize,
    weight_scale: f32,
    bias_scale: f32,
    label: &str,
) -> Result<(LinearLayer, usize)> {
    let weight = Array2::from_shape_fn((out_dim, in_dim), |_| bench_rng_f32(seed, weight_scale));
    let bias = Array1::from_shape_fn(out_dim, |_| bench_rng_f32(seed, bias_scale));
    let layer = LinearLayer::new(weight, Some(bias)).map_err(|e| {
        NyError::InvalidSpec(format!("{label} linear layer shape should be valid: {e}"))
    })?;
    Ok((layer, out_dim * in_dim + out_dim))
}

fn build_random_conv_layer(
    seed: &mut u64,
    spec: ConvBenchSpec,
    weight_scale: f32,
    bias_scale: f32,
    label: &str,
) -> Result<(Conv2dLayer, usize)> {
    let (out_c, in_c, kernel, stride, padding, input_hw) = spec;
    let conv_kernel = ArrayD::from_shape_vec(
        IxDyn(&[out_c, in_c, kernel, kernel]),
        (0..(out_c * in_c * kernel * kernel))
            .map(|_| bench_rng_f32(seed, weight_scale))
            .collect(),
    )
    .map_err(|e| NyError::InvalidSpec(format!("{label} conv kernel shape should be valid: {e}")))?;
    let conv_bias = Array1::from_shape_fn(out_c, |_| bench_rng_f32(seed, bias_scale));
    let layer = Conv2dLayer::with_input_shape(
        conv_kernel,
        Some(conv_bias),
        stride,
        padding,
        input_hw.0,
        input_hw.1,
    )?;
    Ok((layer, out_c * in_c * kernel * kernel + out_c))
}

fn add_conv_sequence(
    network: &mut Network,
    seed: &mut u64,
    conv_specs: &[ConvBenchSpec],
    weight_scale: f32,
    bias_scale: f32,
    skip_final_relu: bool,
    label: &str,
) -> Result<usize> {
    let mut parameter_count = 0usize;

    for (index, &(out_c, in_c, kernel, stride, padding, input_hw)) in conv_specs.iter().enumerate()
    {
        let (layer, params) = build_random_conv_layer(
            seed,
            (out_c, in_c, kernel, stride, padding, input_hw),
            weight_scale,
            bias_scale,
            label,
        )?;
        parameter_count += params;
        network.add_layer(Layer::Conv2d(layer));
        if !(skip_final_relu && index + 1 == conv_specs.len()) {
            network.add_layer(Layer::ReLU(ReLULayer));
        }
    }

    Ok(parameter_count)
}

fn build_mlp_case(
    name: &'static str,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    num_hidden_layers: usize,
) -> Result<BenchCase> {
    let mut seed = 42u64;
    let mut network = Network::new();
    let mut parameter_count = 0usize;

    let first_weight = Array2::from_shape_fn((hidden, in_dim), |_| bench_rng_f32(&mut seed, 0.2));
    let first_bias = Array1::from_shape_fn(hidden, |_| bench_rng_f32(&mut seed, 0.1));
    parameter_count += hidden * in_dim + hidden;
    network.add_layer(Layer::Linear(LinearLayer::new(
        first_weight,
        Some(first_bias),
    )?));

    for _ in 1..num_hidden_layers {
        network.add_layer(Layer::ReLU(ReLULayer));
        let weight = Array2::from_shape_fn((hidden, hidden), |_| bench_rng_f32(&mut seed, 0.2));
        let bias = Array1::from_shape_fn(hidden, |_| bench_rng_f32(&mut seed, 0.1));
        parameter_count += hidden * hidden + hidden;
        network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias))?));
    }

    network.add_layer(Layer::ReLU(ReLULayer));
    let final_weight = Array2::from_shape_fn((out_dim, hidden), |_| bench_rng_f32(&mut seed, 0.2));
    let final_bias = Array1::from_shape_fn(out_dim, |_| bench_rng_f32(&mut seed, 0.1));
    parameter_count += out_dim * hidden + out_dim;
    network.add_layer(Layer::Linear(LinearLayer::new(
        final_weight,
        Some(final_bias),
    )?));

    let eps = 0.01f32;
    let input = BoundedTensor::new(
        Array1::from_shape_fn(in_dim, |i| 0.5 - eps + 0.001 * i as f32).into_dyn(),
        Array1::from_shape_fn(in_dim, |i| 0.5 + eps + 0.001 * i as f32).into_dyn(),
    )?;

    let max_dim = hidden.max(in_dim).max(out_dim);
    let estimated_cpu_peak_bytes = estimate_dense_peak_bytes(max_dim);

    Ok(BenchCase {
        name,
        network,
        input,
        parameter_count,
        estimated_cpu_peak_bytes,
    })
}

fn build_metaroom_case() -> Result<BenchCase> {
    let mut seed = 7u64;
    let mut network = Network::new();
    let mut parameter_count = 0usize;
    parameter_count += add_conv_sequence(
        &mut network,
        &mut seed,
        &METAROOM_CONV_SPECS,
        0.3,
        0.1,
        false,
        "metaroom",
    )?;

    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));

    let hidden_dim = METAROOM_HIDDEN_DIM;
    let flattened = conv_output_dim(*METAROOM_CONV_SPECS.last().expect("metaroom conv stack"));
    let (hidden_layer, hidden_params) =
        build_random_linear_layer(&mut seed, hidden_dim, flattened, 0.15, 0.05, "metaroom")?;
    parameter_count += hidden_params;
    network.add_layer(Layer::Linear(hidden_layer));
    network.add_layer(Layer::ReLU(ReLULayer));

    let output_dim = METAROOM_OUTPUT_DIM;
    let (output_layer, output_params) =
        build_random_linear_layer(&mut seed, output_dim, hidden_dim, 0.15, 0.05, "metaroom")?;
    parameter_count += output_params;
    network.add_layer(Layer::Linear(output_layer));

    let [channels, height, width] = METAROOM_INPUT_SHAPE;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[channels, height, width]),
            vec![-0.25; shape_product3(METAROOM_INPUT_SHAPE)],
        )
        .map_err(|e| {
            NyError::InvalidSpec(format!("metaroom lower input shape should be valid: {e}"))
        })?,
        ArrayD::from_shape_vec(
            IxDyn(&[channels, height, width]),
            vec![0.25; shape_product3(METAROOM_INPUT_SHAPE)],
        )
        .map_err(|e| {
            NyError::InvalidSpec(format!("metaroom upper input shape should be valid: {e}"))
        })?,
    )?;

    let estimated_cpu_peak_bytes =
        estimate_dense_peak_bytes(conv_max_output_dim(&METAROOM_CONV_SPECS));

    Ok(BenchCase {
        name: METAROOM_CASE_NAME,
        network,
        input,
        parameter_count,
        estimated_cpu_peak_bytes,
    })
}

fn build_soundnessbench_case() -> Result<BenchCase> {
    let mut seed = 11u64;
    let mut network = Network::new();
    let mut parameter_count = 0usize;

    let input_dim = SOUNDNESSBENCH_INPUT_DIM;
    let reshape_dim = shape_product3(SOUNDNESSBENCH_RESHAPE_SHAPE);
    let (front_layer, front_params) = build_random_linear_layer(
        &mut seed,
        reshape_dim,
        input_dim,
        0.12,
        0.04,
        "soundnessbench",
    )?;
    parameter_count += front_params;
    network.add_layer(Layer::Linear(front_layer));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(
        SOUNDNESSBENCH_RESHAPE_SHAPE
            .iter()
            .map(|&dim| dim as i64)
            .collect(),
    )));
    parameter_count += add_conv_sequence(
        &mut network,
        &mut seed,
        &SOUNDNESSBENCH_CONV_SPECS,
        0.15,
        0.05,
        true,
        "soundnessbench",
    )?;

    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));

    let output_dim = SOUNDNESSBENCH_OUTPUT_DIM;
    let (output_layer, output_params) = build_random_linear_layer(
        &mut seed,
        output_dim,
        output_dim,
        0.08,
        0.03,
        "soundnessbench",
    )?;
    parameter_count += output_params;
    network.add_layer(Layer::Linear(output_layer));

    let eps = 0.01f32;
    let input = BoundedTensor::new(
        Array1::from_shape_fn(input_dim, |i| 0.5 - eps + 0.001 * i as f32).into_dyn(),
        Array1::from_shape_fn(input_dim, |i| 0.5 + eps + 0.001 * i as f32).into_dyn(),
    )?;

    let estimated_cpu_peak_bytes =
        estimate_dense_peak_bytes(conv_max_output_dim(&SOUNDNESSBENCH_CONV_SPECS));

    Ok(BenchCase {
        name: SOUNDNESSBENCH_CASE_NAME,
        network,
        input,
        parameter_count,
        estimated_cpu_peak_bytes,
    })
}

#[doc(hidden)]
pub fn build_bench_cases() -> Result<[BenchCase; 3]> {
    Ok([
        build_mlp_case("acasxu_like", 5, 50, 5, 6)?,
        build_soundnessbench_case()?,
        build_metaroom_case()?,
    ])
}

#[doc(hidden)]
pub fn clear_gpu_crown_working_set(device: &ComputeDevice) -> Result<()> {
    device.clear_crown_working_set()
}
