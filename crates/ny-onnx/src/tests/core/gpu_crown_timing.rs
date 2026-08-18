// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward timing benchmarks for #3460 acceptance criteria.
//!
//! Separated from gpu_crown.rs to avoid cross-worker file ownership conflicts.
//! These tests measure wall-clock CPU vs GPU CROWN backward timing on real
//! VNN-COMP models to satisfy benchmark acceptance criteria.
//!
//! This module is compiled only with the `benchmarks` feature so missing
//! benchmark checkout or missing GPU hardware becomes an explicit failure for
//! opted-in timing runs instead of a silent green pass.

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_gpu::{Backend, ComputeDevice};
use ny_propagate::Network;
use ny_test_utils::{require_external_model, workspace_root};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Resolve a path relative to the workspace root for benchmark data.
fn benchmark_path(rel: &str) -> PathBuf {
    workspace_root().join(rel)
}

/// Timing benchmarks require a real GPU once the `benchmarks` feature is enabled.
fn require_gpu_device() -> ComputeDevice {
    ComputeDevice::new(Backend::Wgpu).expect(
        "GPU timing benchmarks require a WGPU-compatible device when --features benchmarks is enabled",
    )
}

fn clear_gpu_crown_working_set(device: &ComputeDevice) {
    device
        .clear_crown_working_set()
        .expect("GPU timing benchmark cleanup should succeed");
}

/// Build an epsilon-ball input from the model's input spec (strips batch dim).
fn model_input(model: &OnnxModel, eps: f32) -> BoundedTensor {
    let input_spec = model
        .network
        .inputs
        .first()
        .expect("model has no input spec");
    let shape: Vec<usize> = input_spec.shape[1..]
        .iter()
        .map(|&d| if d > 0 { d as usize } else { 1 })
        .collect();
    let center = ArrayD::zeros(IxDyn(&shape));
    BoundedTensor::from_epsilon(center, eps).expect("BoundedTensor from_epsilon")
}

/// Run CROWN N times and return sorted durations (for median extraction).
fn measure_crown_runs(
    network: &Network,
    input: &BoundedTensor,
    gpu_device: Option<&ComputeDevice>,
    engine: Option<&dyn GemmEngine>,
    n: usize,
) -> Vec<Duration> {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        if let Some(device) = gpu_device {
            clear_gpu_crown_working_set(device);
        }
        let start = Instant::now();
        let _ = match engine {
            Some(e) => network
                .propagate_crown_with_engine(input, Some(e))
                .expect("CROWN failed"),
            None => network.propagate_crown(input).expect("CROWN failed"),
        };
        times.push(start.elapsed());
        if let Some(device) = gpu_device {
            clear_gpu_crown_working_set(device);
        }
    }
    times.sort();
    times
}

/// Compute max element-wise difference between GPU and CPU bounds.
fn max_bound_diff(gpu: &BoundedTensor, cpu: &BoundedTensor) -> f32 {
    let cpu_lo = cpu.lower().as_slice().expect("cpu lower");
    let cpu_hi = cpu.upper().as_slice().expect("cpu upper");
    let gpu_lo = gpu.lower().as_slice().expect("gpu lower");
    let gpu_hi = gpu.upper().as_slice().expect("gpu upper");
    assert_eq!(cpu_lo.len(), gpu_lo.len(), "output dim mismatch");

    let mut max_diff: f32 = 0.0;
    for i in 0..cpu_lo.len() {
        let lo_diff = (gpu_lo[i] - cpu_lo[i]).abs();
        let hi_diff = (gpu_hi[i] - cpu_hi[i]).abs();
        max_diff = max_diff.max(lo_diff).max(hi_diff);
    }
    max_diff
}

/// Print timing report for GPU vs CPU CROWN benchmark.
fn print_timing_report(
    label: &str,
    num_layers: usize,
    output_dim: usize,
    cpu_times: &[Duration],
    gpu_times: &[Duration],
    max_diff: f32,
) {
    let cpu_median = cpu_times[cpu_times.len() / 2];
    let gpu_median = gpu_times[gpu_times.len() / 2];
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("{label} GPU vs CPU CROWN backward timing (#3460)");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("Layers: {num_layers}, Output dim: {output_dim}");
    eprintln!(
        "CPU CROWN (median): {:.3}ms  runs: {:.3}ms, {:.3}ms, {:.3}ms",
        cpu_median.as_secs_f64() * 1000.0,
        cpu_times[0].as_secs_f64() * 1000.0,
        cpu_times[1].as_secs_f64() * 1000.0,
        cpu_times[2].as_secs_f64() * 1000.0,
    );
    eprintln!(
        "GPU CROWN (median): {:.3}ms  runs: {:.3}ms, {:.3}ms, {:.3}ms",
        gpu_median.as_secs_f64() * 1000.0,
        gpu_times[0].as_secs_f64() * 1000.0,
        gpu_times[1].as_secs_f64() * 1000.0,
        gpu_times[2].as_secs_f64() * 1000.0,
    );
    let ratio = if gpu_median.as_nanos() > 0 {
        cpu_median.as_secs_f64() / gpu_median.as_secs_f64()
    } else {
        f64::NAN
    };
    eprintln!("Speedup: {ratio:.2}x | Max bound diff: {max_diff:.2e}");
    eprintln!("═══════════════════════════════════════════════════════════");
}

// ───────────────────────────────────────────────────────────────────────
// ACAS-Xu GPU vs CPU CROWN backward timing (#3460 acceptance criterion)
// ───────────────────────────────────────────────────────────────────────

/// Timing benchmark: ACAS-Xu 1_1 CPU CROWN vs GPU CROWN backward.
///
/// ACAS-Xu is small (5 inputs, 6x50 hidden, 5 outputs, 22 layers with
/// AddConstant/SubConstant). CPU CROWN is fast (<300ms), so this benchmark
/// validates GPU extraction correctness rather than speedup. CPU and GPU use
/// different sound relaxation slopes, so their raw bounds are reported but are
/// not expected to agree to floating-point noise.
#[ntest::timeout(120000)]
#[test]
fn test_gpu_crown_acasxu_timing_benchmark_3460() {
    let model_path = benchmark_path(
        "benchmarks/vnncomp2023/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
    );
    require_external_model(&model_path);
    let gpu_device = require_gpu_device();

    let model =
        load_onnx(&model_path).unwrap_or_else(|e| panic!("Failed to load ACAS-Xu model: {e}"));
    let network = model
        .to_propagate_network()
        .expect("to_propagate_network failed");
    let input = model_input(&model, 0.01);
    let engine: &dyn GemmEngine = &gpu_device;

    // Warm-up (first GPU call initializes wgpu device and shaders).
    clear_gpu_crown_working_set(&gpu_device);
    let _ = network.propagate_crown_with_engine(&input, Some(engine));
    clear_gpu_crown_working_set(&gpu_device);

    let cpu_times = measure_crown_runs(&network, &input, None, None, 3);
    let gpu_times = measure_crown_runs(&network, &input, Some(&gpu_device), Some(engine), 3);

    let cpu_crown = network.propagate_crown(&input).expect("CPU CROWN");
    clear_gpu_crown_working_set(&gpu_device);
    let gpu_crown = network
        .propagate_crown_with_engine(&input, Some(engine))
        .expect("GPU CROWN");
    clear_gpu_crown_working_set(&gpu_device);
    let diff = max_bound_diff(&gpu_crown, &cpu_crown);
    let output_dim = cpu_crown.lower().as_slice().expect("lo").len();

    print_timing_report(
        "ACAS-Xu 1_1",
        network.layers().len(),
        output_dim,
        &cpu_times,
        &gpu_times,
        diff,
    );

    // CPU f64-certified and GPU directed-f32 CROWN collect their own IBP
    // intermediates and can legitimately select different ReLU slopes. Exact
    // parity is therefore the wrong correctness criterion (the canonical ACAS
    // regression documents a stable ~4.7e-2 delta). Require the timed GPU result
    // to satisfy the actual soundness gates instead: finite ordered bounds, no
    // looser than IBP, and enclosure of concrete outputs over the input box.
    let ibp = network.propagate_ibp(&input).expect("IBP");
    gpu_crown::assert_crown_finite_within_ibp(&gpu_crown, &ibp, "ACAS-Xu 1_1 timed GPU");
    gpu_crown::assert_crown_encloses_acas_samples(
        &network,
        &input,
        &gpu_crown,
        512,
        "ACAS-Xu 1_1 timed GPU",
    );
}
