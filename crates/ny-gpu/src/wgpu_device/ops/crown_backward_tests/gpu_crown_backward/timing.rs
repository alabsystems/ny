// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Competition-shape GPU CROWN backward timing tests (#3397).
//!
//! Builds networks matching metaroom_6cnn_ry and soundnessbench_exact shapes,
//! then measures GPU backward pass time. These tests validate that the GPU path
//! meets the VNN-COMP 180s timeout budget.

use super::*;
use crate::wgpu_device::{estimate_crown_backward_peak_bytes, gpu_memory_budget_bytes};
// Blessed env-mutation choke point (clippy env wall): replaces the previous
// local ScopedEnvVar + lock duplicates.
use ny_test_utils::env::ScopedEnvVar;

use crate::benchmark_support::crown_backward_workloads::{
    bench_rng_f32, conv_output_dim, shape_product3, ConvBenchSpec, METAROOM_CASE_NAME,
    METAROOM_CONV_SPECS, METAROOM_HIDDEN_DIM, METAROOM_INPUT_SHAPE, METAROOM_OUTPUT_DIM,
    SOUNDNESSBENCH_CASE_NAME, SOUNDNESSBENCH_CONV_SPECS, SOUNDNESSBENCH_INPUT_DIM,
    SOUNDNESSBENCH_OUTPUT_DIM, SOUNDNESSBENCH_RESHAPE_SHAPE,
};

/// Hard ceiling for timing tests even when production budget overrides are larger.
const MAX_TIMING_TEST_MEMORY_BYTES: usize = 16 * 1024 * 1024 * 1024;

fn timing_test_memory_budget_bytes() -> usize {
    gpu_memory_budget_bytes().min(MAX_TIMING_TEST_MEMORY_BYTES)
}

/// Check that estimated memory is within the production GPU budget before running
/// a timing test. Reuses the production helper, but preserves a 16 GiB hard cap
/// so local env overrides cannot re-enable the high-RSS timing runs that #3515
/// was meant to prevent.
fn check_memory_budget(label: &str, layers: &[GpuCrownLayer], num_specs: usize) {
    let estimated = estimate_crown_backward_peak_bytes(layers, num_specs);
    let budget = timing_test_memory_budget_bytes();
    assert!(
        estimated <= budget,
        "{label}: estimated peak memory {:.1} GB exceeds {:.1} GB budget. \
         Reduce num_specs or layer dimensions to prevent OOM (#3515).",
        estimated as f64 / (1024.0 * 1024.0 * 1024.0),
        budget as f64 / (1024.0 * 1024.0 * 1024.0),
    );
}

/// Build a stack of Conv2d + ReLU layers in backward order from forward specs.
///
/// Appends to `layers`. The caller handles the activation immediately above the
/// last forward conv, so this helper only inserts ReLUs between successive convs.
fn build_conv_stack(
    layers: &mut Vec<GpuCrownLayer>,
    conv_specs: &[ConvBenchSpec],
    seed: &mut u64,
    weight_scale: f32,
) {
    for (idx, &(oc, ic, k, (stride_h, stride_w), (pad_h, pad_w), (in_h, in_w))) in
        conv_specs.iter().rev().enumerate()
    {
        let out_h = (in_h + 2 * pad_h - k) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - k) / stride_w + 1;
        let kernel_cols = ic * k * k;
        let flat_out = oc * out_h * out_w;

        if idx > 0 {
            layers.push(GpuCrownLayer::Activation {
                lower_slope: vec![0.5; flat_out],
                upper_slope: vec![0.5; flat_out],
                lower_intercept: vec![0.0; flat_out],
                upper_intercept: vec![0.1; flat_out],
                num_neurons: flat_out,
            });
        }

        let weight_col: Vec<f32> = (0..oc * kernel_cols)
            .map(|_| bench_rng_f32(seed, weight_scale))
            .collect();
        let bias_expanded: Vec<f32> = (0..flat_out).map(|_| bench_rng_f32(seed, 0.05)).collect();

        layers.push(GpuCrownLayer::Conv2d {
            weight_col: weight_col.into(),
            bias_expanded: Some(bias_expanded.into()),
            out_channels: oc,
            in_channels: ic,
            kernel_h: k,
            kernel_w: k,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            cert_err: Default::default(),
        });
    }
}

/// Push a Linear layer with random weights onto `layers`.
fn push_linear(
    layers: &mut Vec<GpuCrownLayer>,
    seed: &mut u64,
    out_f: usize,
    in_f: usize,
    w_scale: f32,
) {
    let w: Vec<f32> = (0..out_f * in_f)
        .map(|_| bench_rng_f32(seed, w_scale))
        .collect();
    let b: Vec<f32> = (0..out_f).map(|_| bench_rng_f32(seed, 0.05)).collect();
    layers.push(GpuCrownLayer::Linear {
        weight: w.into(),
        bias: Some(b.into()),
        out_features: out_f,
        in_features: in_f,
        cert_err: Default::default(),
    });
}

/// Push a ReLU activation layer onto `layers`.
fn push_relu(layers: &mut Vec<GpuCrownLayer>, dim: usize) {
    layers.push(GpuCrownLayer::Activation {
        lower_slope: vec![0.5; dim],
        upper_slope: vec![0.5; dim],
        lower_intercept: vec![0.0; dim],
        upper_intercept: vec![0.1; dim],
        num_neurons: dim,
    });
}

/// Build metaroom_6cnn_ry-shaped layers: 4 conv + 2 FC, 20 specs, 5376 input.
pub(super) fn build_metaroom_like_layers() -> (Vec<GpuCrownLayer>, usize, usize) {
    let mut seed = 42u64;
    let num_specs = METAROOM_OUTPUT_DIM;
    let input_dim = shape_product3(METAROOM_INPUT_SHAPE);
    let mut layers = Vec::new();
    let flat = conv_output_dim(*METAROOM_CONV_SPECS.last().expect("metaroom conv stack"));

    push_linear(&mut layers, &mut seed, num_specs, METAROOM_HIDDEN_DIM, 0.15);
    push_relu(&mut layers, METAROOM_HIDDEN_DIM);
    push_linear(&mut layers, &mut seed, METAROOM_HIDDEN_DIM, flat, 0.1);
    push_relu(&mut layers, flat);
    build_conv_stack(&mut layers, &METAROOM_CONV_SPECS, &mut seed, 0.2);

    (layers, input_dim, num_specs)
}

/// Build soundnessbench-shaped layers: Linear + 6 Conv2d + Linear, 384 specs.
pub(super) fn build_soundnessbench_like_layers() -> (Vec<GpuCrownLayer>, usize, usize) {
    let mut seed = 77u64;
    let num_specs = SOUNDNESSBENCH_OUTPUT_DIM;
    let input_dim = SOUNDNESSBENCH_INPUT_DIM;
    let mut layers = Vec::new();
    let reshape_dim = shape_product3(SOUNDNESSBENCH_RESHAPE_SHAPE);

    push_linear(&mut layers, &mut seed, num_specs, num_specs, 0.08);
    build_conv_stack(&mut layers, &SOUNDNESSBENCH_CONV_SPECS, &mut seed, 0.15);
    push_relu(&mut layers, reshape_dim);
    push_linear(&mut layers, &mut seed, reshape_dim, input_dim, 0.12);

    (layers, input_dim, num_specs)
}

/// Run a timing test: warmup + timed run, assert lower <= upper for all specs.
fn run_timing_test(
    label: &str,
    layers: &[GpuCrownLayer],
    num_specs: usize,
    inp_l: &[f32],
    inp_u: &[f32],
) {
    let device = require_device();
    let spec = identity_spec(num_specs);

    device
        .clear_crown_working_set()
        .expect("clear CROWN working set before timing test");

    // Warmup (populates plan cache, JIT shader compilation)
    let _ = device
        .crown_backward_gpu(layers, &spec, num_specs, inp_l, inp_u)
        .expect("warmup should succeed");

    device
        .clear_crown_working_set()
        .expect("clear CROWN working set after warmup");

    let start = std::time::Instant::now();
    let result = device
        .crown_backward_gpu(layers, &spec, num_specs, inp_l, inp_u)
        .expect("timed run should succeed");
    let elapsed = start.elapsed();

    // Assert timing was captured (non-zero elapsed)
    assert!(elapsed.as_millis() > 0, "{label}: elapsed was zero");

    // Soundness: lower <= upper for all specs
    for i in 0..num_specs {
        assert!(
            result.lower_bounds[i] <= result.upper_bounds[i] + 1e-3,
            "{label} spec {i}: lower {} > upper {}",
            result.lower_bounds[i],
            result.upper_bounds[i]
        );
    }
}

pub(super) fn run_manual_spec_batches(
    device: &WgpuDevice,
    layers: &[GpuCrownLayer],
    spec: &[f32],
    num_specs: usize,
    batch_size: usize,
    inp_l: &[f32],
    inp_u: &[f32],
) -> GpuCrownResult {
    assert!(batch_size > 0, "manual batch size must be positive");
    let first_dim = num_specs;
    let mut lower_bounds = Vec::with_capacity(num_specs);
    let mut upper_bounds = Vec::with_capacity(num_specs);
    let mut spec_offset = 0usize;
    let mut row_offset = 0usize;

    while row_offset < num_specs {
        let batch_specs = batch_size.min(num_specs - row_offset);
        let batch_elems = batch_specs * first_dim;
        let batch_spec = &spec[spec_offset..spec_offset + batch_elems];
        let batch_result = device
            .crown_backward_gpu(layers, batch_spec, batch_specs, inp_l, inp_u)
            .expect("manual spec batch should succeed");
        lower_bounds.extend_from_slice(&batch_result.lower_bounds);
        upper_bounds.extend_from_slice(&batch_result.upper_bounds);
        row_offset += batch_specs;
        spec_offset += batch_elems;
    }

    GpuCrownResult {
        lower_bounds,
        upper_bounds,
    }
}

pub(super) fn assert_results_close(
    actual: &GpuCrownResult,
    expected: &GpuCrownResult,
    eps: f32,
    label: &str,
) {
    assert_eq!(
        actual.lower_bounds.len(),
        expected.lower_bounds.len(),
        "{label}: lower result length mismatch"
    );
    assert_eq!(
        actual.upper_bounds.len(),
        expected.upper_bounds.len(),
        "{label}: upper result length mismatch"
    );

    for (idx, (&actual_lower, &expected_lower)) in actual
        .lower_bounds
        .iter()
        .zip(expected.lower_bounds.iter())
        .enumerate()
    {
        let diff = (actual_lower - expected_lower).abs();
        assert!(
            diff <= eps,
            "{label}: lower[{idx}] mismatch actual={actual_lower} expected={expected_lower} diff={diff} eps={eps}"
        );
    }

    for (idx, (&actual_upper, &expected_upper)) in actual
        .upper_bounds
        .iter()
        .zip(expected.upper_bounds.iter())
        .enumerate()
    {
        let diff = (actual_upper - expected_upper).abs();
        assert!(
            diff <= eps,
            "{label}: upper[{idx}] mismatch actual={actual_upper} expected={expected_upper} diff={diff} eps={eps}"
        );
    }
}

/// Metaroom 6cnn_ry: 4 conv + FC, 20 specs. CPU >900s, GPU <1s.
#[test]
fn test_crown_backward_gpu_metaroom_like_timing() {
    let _gpu_serial = gpu_test_serial_guard();
    let (layers, input_dim, num_specs) = build_metaroom_like_layers();
    check_memory_budget(METAROOM_CASE_NAME, &layers, num_specs);
    let inp_l = vec![-0.25f32; input_dim];
    let inp_u = vec![0.25f32; input_dim];
    run_timing_test(METAROOM_CASE_NAME, &layers, num_specs, &inp_l, &inp_u);
}

/// Soundnessbench: Linear + 6 conv + Linear, 384 specs. CPU 915s, GPU <2s.
#[test]
fn test_crown_backward_gpu_soundnessbench_like_timing() {
    let _gpu_serial = gpu_test_serial_guard();
    let (layers, input_dim, num_specs) = build_soundnessbench_like_layers();
    check_memory_budget(SOUNDNESSBENCH_CASE_NAME, &layers, num_specs);
    let inp_l = vec![-0.01f32; input_dim];
    let inp_u = vec![0.01f32; input_dim];
    run_timing_test(SOUNDNESSBENCH_CASE_NAME, &layers, num_specs, &inp_l, &inp_u);
}

/// Regression for #3397 spec batching: the competition-shape soundnessbench
/// workload must produce the same bounds whether specs are run in one
/// auto-batched invocation or in explicit manual chunks.
#[test]
fn test_crown_backward_gpu_soundnessbench_spec_batching_matches_manual_chunks() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let (layers, input_dim, num_specs) = build_soundnessbench_like_layers();
    check_memory_budget(SOUNDNESSBENCH_CASE_NAME, &layers, num_specs);
    let inp_l = vec![-0.01f32; input_dim];
    let inp_u = vec![0.01f32; input_dim];
    let spec = identity_spec(num_specs);

    let auto_batched = device
        .crown_backward_gpu(&layers, &spec, num_specs, &inp_l, &inp_u)
        .expect("auto-batched competition-shape run should succeed");
    device
        .clear_crown_working_set()
        .expect("clear CROWN working set between batching checks");
    let manual_chunks =
        run_manual_spec_batches(&device, &layers, &spec, num_specs, 96, &inp_l, &inp_u);

    assert_results_close(
        &auto_batched,
        &manual_chunks,
        1e-4,
        "soundnessbench spec batching",
    );
}

#[test]
fn test_crown_backward_gpu_soundnessbench_timestamp_profile_reports_gemm_3599() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    assert!(
        device.supports_timestamp_queries(),
        "gpu-tests timestamp conformance requires an adapter with timestamp queries"
    );

    let (layers, input_dim, num_specs) = build_soundnessbench_like_layers();
    check_memory_budget(SOUNDNESSBENCH_CASE_NAME, &layers, num_specs);
    let inp_l = vec![-0.01f32; input_dim];
    let inp_u = vec![0.01f32; input_dim];
    let spec = identity_spec(num_specs);

    device
        .set_crown_timestamp_profiling(true)
        .expect("enable GPU CROWN timestamp profiling");
    let profile_result = {
        let _ = device
            .take_last_crown_timestamp_profile()
            .expect("clear stale timestamp profile");
        device
            .crown_backward_gpu(&layers, &spec, num_specs, &inp_l, &inp_u)
            .expect("profiled soundnessbench run should succeed");
        device
            .take_last_crown_timestamp_profile()
            .expect("read timestamp profile")
            .ok_or_else(|| "missing GPU CROWN timestamp profile".to_string())
    };
    device
        .set_crown_timestamp_profiling(false)
        .expect("disable GPU CROWN timestamp profiling");

    let profile = profile_result.expect("profiled soundnessbench run should produce timestamps");
    assert!(
        profile.total_seconds() > 0.0,
        "profiled soundnessbench run should report positive GPU time"
    );
    let summaries = profile.summarize_by_label();

    let lower = summaries
        .iter()
        .find(|summary| summary.label == "crown_gemm_lower")
        .expect("profile should include lower GEMM passes");
    assert!(
        lower.total_seconds > 0.0,
        "lower GEMM passes should have positive GPU time"
    );

    let upper = summaries
        .iter()
        .find(|summary| summary.label == "crown_gemm_upper")
        .expect("profile should include upper GEMM passes");
    assert!(
        upper.total_seconds > 0.0,
        "upper GEMM passes should have positive GPU time"
    );
}

#[test]
fn test_timing_budget_keeps_16gb_safety_cap_3515() {
    let _guard = ny_test_utils::env::lock_env();
    let _budget = ScopedEnvVar::set("NY_GPU_MEMORY_BUDGET_MB", "65536");
    assert_eq!(
        timing_test_memory_budget_bytes(),
        MAX_TIMING_TEST_MEMORY_BYTES,
        "timing tests must keep the 16 GiB hard cap even when production overrides are larger",
    );
}
