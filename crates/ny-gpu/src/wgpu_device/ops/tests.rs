// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU integration tests for wgpu device operations.
//!
//! These tests require a GPU and are gated behind the `gpu-tests` feature.
//! Run with: `cargo test -p ny-gpu --features gpu-tests`
//!
//! Without the feature, these tests are not compiled — preventing silent
//! green passes on CI machines without GPU hardware.

#[cfg(feature = "gpu-tests")]
use super::super::WgpuDevice;
#[cfg(feature = "gpu-tests")]
use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};
#[cfg(feature = "gpu-tests")]
use crate::{wgpu_device::params::ScaleIbpParams, FALLBACK_BOUND};
#[cfg(feature = "gpu-tests")]
use approx::assert_relative_eq;
#[cfg(feature = "gpu-tests")]
use ny_core::{
    GpuDagIbpForwardExt, GpuDagIbpOp, GpuDagIbpPlanDesc, GpuIbpForward, GpuIbpLayer,
    NETWORK_INPUT_IDX,
};
#[cfg(feature = "gpu-tests")]
use ny_propagate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
#[cfg(feature = "gpu-tests")]
use ny_propagate::{Layer, Network};
#[cfg(feature = "gpu-tests")]
use ny_test_utils::{assert_bounded_tensor_close, GPU_REGRESSION_RELAXED_EPSILON};
#[cfg(feature = "gpu-tests")]
use rayon::prelude::*;
#[cfg(feature = "gpu-tests")]
use std::sync::Arc;
#[cfg(feature = "gpu-tests")]
use std::time::Instant;

#[cfg(feature = "gpu-tests")]
fn assert_bounds_valid(output: &ny_tensor::BoundedTensor) {
    let lower = output.lower();
    let upper = output.upper();

    assert_eq!(lower.shape(), upper.shape());
    assert_eq!(lower.len(), upper.len());

    for (idx, (l, u)) in lower.iter().zip(upper.iter()).enumerate() {
        assert!(l.is_finite(), "lower[{}] not finite: {}", idx, l);
        assert!(u.is_finite(), "upper[{}] not finite: {}", idx, u);
        assert!(
            l <= u,
            "invalid bounds at {}: lower={} > upper={}",
            idx,
            l,
            u
        );
    }
}

#[cfg(feature = "gpu-tests")]
fn run_scale_ibp_shader_for_test(
    device: &WgpuDevice,
    input_lower: &[f32],
    input_upper: &[f32],
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(input_lower.len(), input_upper.len());
    let total_elements = input_lower.len();
    let total_elements_u32 =
        u32::try_from(total_elements).expect("test helper requires total_elements <= u32::MAX");
    let byte_size = size_of_val(input_lower) as u64;

    let params = ScaleIbpParams {
        total_elements: total_elements_u32,
        scale,
        _padding: [0, 0],
    };

    let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_params_buffer"),
        size: size_of::<ScaleIbpParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input_lower_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_input_lower_buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input_upper_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_input_upper_buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_lower_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_output_lower_buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_upper_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_output_upper_buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_lower = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_staging_lower"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_upper = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scale_staging_upper"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    device
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
    device
        .queue
        .write_buffer(&input_lower_buffer, 0, bytemuck::cast_slice(input_lower));
    device
        .queue
        .write_buffer(&input_upper_buffer, 0, bytemuck::cast_slice(input_upper));

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_scale_bind_group"),
        layout: &device.scale_ibp_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_lower_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input_upper_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_lower_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_upper_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_scale_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test_scale_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&device.scale_ibp_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(total_elements_u32.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_lower_buffer, 0, &staging_lower, 0, byte_size);
    encoder.copy_buffer_to_buffer(&output_upper_buffer, 0, &staging_upper, 0, byte_size);
    device.queue.submit(std::iter::once(encoder.finish()));

    let lower = WgpuDevice::read_buffer(&device.device, &staging_lower, total_elements)
        .expect("scale shader readback lower should succeed");
    let upper = WgpuDevice::read_buffer(&device.device, &staging_upper, total_elements)
        .expect("scale shader readback upper should succeed");
    (lower, upper)
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_device_creation() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let info = device.info();
    // Device info should be a non-empty string describing the GPU
    assert!(!info.is_empty(), "Device info should not be empty");
    // Device should report a valid backend
    assert!(
        info.contains("Metal")
            || info.contains("Vulkan")
            || info.contains("Dx12")
            || info.contains("Gl"),
        "Device info should contain a recognized backend name, got: {}",
        info
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_linear_ibp_nan_guard_zero_times_inf() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Input includes infinities so wp=0 can trigger 0*inf in shader arithmetic.
    let input_lower = ndarray::array![[0.0f32]].into_dyn();
    let input_upper = ndarray::array![[f32::INFINITY]].into_dyn();
    let input = ny_tensor::BoundedTensor::new_unchecked(input_lower, input_upper).unwrap();

    // Negative weight => wp = max(w, 0) = 0, so wp * x can produce 0*inf.
    let weight = ndarray::array![[-1.0f32]];

    let output = device.linear_ibp(&input, &weight, None).unwrap();
    assert_eq!(output.shape(), vec![1, 1]);
    assert_bounds_valid(&output);

    let lower = output.lower()[[0, 0]];
    let upper = output.upper()[[0, 0]];
    assert!(
        lower <= -0.99 * FALLBACK_BOUND,
        "expected conservative lower widening for 0*inf guard, got [{lower}, {upper}]"
    );
    assert!(
        upper >= -1.0e-6,
        "expected non-negative upper bound for x in [0, +inf] with y=-x, got [{lower}, {upper}]"
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_matmul_ibp_nan_guard_zero_times_inf() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // A is exactly zero, B spans infinities => corner products include 0*inf.
    let a_lower = ndarray::array![[0.0f32]].into_dyn();
    let a_upper = ndarray::array![[0.0f32]].into_dyn();
    let input_a = ny_tensor::BoundedTensor::new(a_lower, a_upper).unwrap();

    let b_lower = ndarray::array![[-f32::INFINITY]].into_dyn();
    let b_upper = ndarray::array![[f32::INFINITY]].into_dyn();
    let input_b = ny_tensor::BoundedTensor::new_unchecked(b_lower, b_upper).unwrap();

    let output = device.matmul_ibp(&input_a, &input_b).unwrap();
    assert_eq!(output.shape(), vec![1, 1]);
    assert_bounds_valid(&output);

    let lower = output.lower()[[0, 0]];
    let upper = output.upper()[[0, 0]];
    assert!(
        lower <= -0.99 * FALLBACK_BOUND && upper >= 0.99 * FALLBACK_BOUND,
        "expected conservative widening for 0*inf guard, got [{lower}, {upper}]"
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_gemm_f32_basic() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let m = 2usize;
    let k = 3usize;
    let n = 4usize;

    // A: 2x3
    let a = vec![
        1.0f32, 2.0, 3.0, //
        -1.0, 0.5, 2.0, //
    ];
    // B: 3x4
    let b = vec![
        0.25f32, -1.0, 2.0, 0.0, //
        1.5, 0.5, -0.5, 1.0, //
        2.0, 1.0, 0.0, -2.0, //
    ];

    let out = device.gemm_f32(m, k, n, &a, &b).unwrap();
    assert_eq!(out.len(), m * n);

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[row * k + t] * b[t * n + col];
            }
            expected[row * n + col] = sum;
        }
    }

    for i in 0..(m * n) {
        assert_relative_eq!(out[i], expected[i], epsilon = 1e-4);
    }
}

/// `gemm_interval_sound` must produce a sound enclosure when its underlying
/// `gemm_f32` runs on the real GPU (round-to-nearest, GPU reduction order).
///
/// This is the device-level proof that the midpoint–radius enclosure
/// (validated against an exact oracle for the CPU engine in ny-core) stays
/// sound on Metal/wgpu — i.e. the GPU can produce CROWN coefficient bounds that
/// never exclude a reachable product. Compares against an exact f64 corner-sum
/// oracle (`f32·f32` is exact in `f64`).
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_gemm_interval_sound_encloses_true_product_on_gpu() {
    use ny_core::GemmEngine;
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // A few shapes incl. a larger K (more accumulation → larger rounding term).
    for &(m, k, n) in &[(2usize, 3usize, 4usize), (5, 17, 6), (3, 64, 3)] {
        // Deterministic LCG so the test is reproducible without a rand dep.
        let mut state: u64 = 0x1234_5678_9abc_def0 ^ ((m * 131 + k * 17 + n) as u64);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
        };
        let mk = m * k;
        let kn = k * n;
        let (mut a_lo, mut a_hi) = (vec![0.0f32; mk], vec![0.0f32; mk]);
        let (mut b_lo, mut b_hi) = (vec![0.0f32; kn], vec![0.0f32; kn]);
        for i in 0..mk {
            let c = next() * 4.0;
            let w = (next() * 0.5).abs();
            a_lo[i] = c - w;
            a_hi[i] = c + w;
        }
        for i in 0..kn {
            let c = next() * 4.0;
            let w = (next() * 0.5).abs();
            b_lo[i] = c - w;
            b_hi[i] = c + w;
        }

        let diagnostic = super::gemm::WgpuDiagnosticGemm::new(device.as_ref());
        let (c_lo, c_hi) = diagnostic
            .gemm_interval_sound(m, k, n, &a_lo, &a_hi, &b_lo, &b_hi)
            .expect("gpu interval gemm");

        for i in 0..m {
            for j in 0..n {
                let (mut o_lo, mut o_hi) = (0.0f64, 0.0f64);
                for l in 0..k {
                    let al = f64::from(a_lo[i * k + l]);
                    let ah = f64::from(a_hi[i * k + l]);
                    let bl = f64::from(b_lo[l * n + j]);
                    let bh = f64::from(b_hi[l * n + j]);
                    let p = [al * bl, al * bh, ah * bl, ah * bh];
                    o_lo += p.iter().copied().fold(f64::INFINITY, f64::min);
                    o_hi += p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                }
                let idx = i * n + j;
                assert!(
                    f64::from(c_lo[idx]) <= o_lo && f64::from(c_hi[idx]) >= o_hi,
                    "UNSOUND on GPU (m={m},k={k},n={n}) idx {idx}: \
                     true range [{o_lo}, {o_hi}] not in [{}, {}]",
                    c_lo[idx],
                    c_hi[idx]
                );
            }
        }
    }
}

/// `crown_aw_error_step` (per-layer sound coefficient-error propagation, the core
/// of the GPU-resident sound CROWN backward) must stay sound when its three GEMMs
/// run on the real GPU. `a_new ± a_err_new` must bracket every exact `a_exact@w`
/// for `a_exact ∈ [a − a_err, a + a_err]`.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_crown_aw_error_step_sound_on_gpu() {
    use ny_core::GemmEngine;
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    for &(m, k, n) in &[(2usize, 7usize, 3usize), (4, 33, 5), (3, 128, 2)] {
        let mut state: u64 = 0x51A7_E001 ^ ((m * 7 + k * 3 + n) as u64);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| next() * 3.0).collect();
        let a_err: Vec<f32> = (0..m * k).map(|_| (next() * 0.2).abs()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| next() * 3.0).collect();

        let diagnostic = super::gemm::WgpuDiagnosticGemm::new(device.as_ref());
        let (a_new, a_err_new) = diagnostic
            .crown_aw_error_step(m, k, n, &a, &a_err, &w)
            .expect("gpu aw error step");

        for i in 0..m {
            for j in 0..n {
                let (mut tmin, mut tmax) = (0.0f64, 0.0f64);
                for l in 0..k {
                    let amin = f64::from(a[i * k + l]) - f64::from(a_err[i * k + l]);
                    let amax = f64::from(a[i * k + l]) + f64::from(a_err[i * k + l]);
                    let wv = f64::from(w[l * n + j]);
                    tmin += (amin * wv).min(amax * wv);
                    tmax += (amin * wv).max(amax * wv);
                }
                let idx = i * n + j;
                let lo = f64::from(a_new[idx]) - f64::from(a_err_new[idx]);
                let hi = f64::from(a_new[idx]) + f64::from(a_err_new[idx]);
                assert!(
                    lo <= tmin && hi >= tmax,
                    "UNSOUND on GPU ({m}x{k}x{n}) [{i},{j}]: true [{tmin},{tmax}] not in [{lo},{hi}]"
                );
            }
        }
    }
}

/// Regression: parallel GEMM calls on one device must not recycle the shared
/// staging buffer while an earlier readback is still mapped.
///
/// #3813 row 151 hit:
/// `Queue::submit -> Buffer with 'gemm_staging_buffer' label is still mapped`
/// from Rayon-parallel Conv2d transpose GEMM calls under the WGPU relusplitter
/// path. This test exercises the same shared-device pattern directly.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_gemm_f32_parallel_shared_device_reuses_staging_safely_3813() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let m = 128usize;
    let k = 64usize;
    let n = 64usize;

    let a: Vec<f32> = (0..(m * k))
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
        .collect();
    let b: Vec<f32> = (0..(k * n))
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.2)
        .collect();

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[row * k + t] * b[t * n + col];
            }
            expected[row * n + col] = sum;
        }
    }

    let results: Vec<Vec<f32>> = (0..8)
        .into_par_iter()
        .map(|_| {
            let mut last = Vec::new();
            for _ in 0..4 {
                last = device
                    .gemm_f32(m, k, n, &a, &b)
                    .expect("parallel GEMM should not hit mapped staging-buffer panic");
            }
            last
        })
        .collect();

    for out in results {
        assert_eq!(out.len(), expected.len());
        for (actual, expected) in out.iter().zip(expected.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-4);
        }
    }
}

/// Verify that explicit NaN inputs remain downstream-detectable after GEMM (#2708).
/// A corrupted CROWN A-matrix row must not collapse to an ordinary finite value
/// such as 0.0. The later concretize step detects either raw NaN or the exact
/// `FALLBACK_BOUND` sentinel and degrades the row to maximally loose bounds.
/// Accepting an arbitrary bounded finite output here would re-allow the
/// pre-#2708 bug where NaN was rewritten away and a corrupted coefficient
/// contribution was silently dropped.
///
/// This test isolates the NaN case. Finite overflow and fallback-sentinel
/// handling are covered by the separate shader/seeded regressions for #2708.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_gemm_f32_nan_input_preserves_detectable_corruption_2708() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let m = 2usize;
    let k = 3usize;
    let n = 2usize;

    // Row 0 contains an explicit NaN. With finite weights, every output in that
    // row must remain detectably corrupted instead of being zeroed away.
    let a = vec![
        f32::NAN,
        1.0f32,
        2.0, // row 0: NaN, normal, normal
        1.0f32,
        2.0,
        3.0, // row 1: all normal
    ];
    // B is a normal weight matrix
    let b = vec![
        1.0f32, 0.5, //
        2.0, 1.0, //
        0.5, 0.25, //
    ];

    let out = device
        .gemm_f32(m, k, n, &a, &b)
        .expect("GEMM should succeed even with NaN inputs");
    assert_eq!(out.len(), m * n);

    // Row 1 (all-normal inputs) should compute correctly
    // Expected: [1*1 + 2*2 + 3*0.5, 1*0.5 + 2*1 + 3*0.25] = [6.5, 3.25]
    assert_relative_eq!(out[2], 6.5, epsilon = 1e-4);
    assert_relative_eq!(out[3], 3.25, epsilon = 1e-4);

    // Row 0 must stay detectable to the downstream concretize shader: either as
    // raw NaN or as the exact fallback sentinel magnitude. An ordinary finite
    // value (especially 0.0) would hide the corruption and recreate the old bug.
    for i in 0..n {
        let value = out[i];
        assert!(
            value.is_nan() || value.abs().to_bits() == FALLBACK_BOUND.to_bits(),
            "corrupted GEMM output[{i}] = {value} must remain NaN or exact FALLBACK_BOUND"
        );
    }
}

/// Verify that GEMM clamping handles 0 * Inf (common source of NaN in CROWN backward).
/// When a prior layer produces Inf bounds and the current layer has a zero weight,
/// the product 0 * Inf = NaN. The shader must catch this.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_gemm_f32_zero_times_inf() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // 1x2 @ 2x1: single dot product of [0, Inf] . [anything, anything]
    let a = vec![0.0f32, f32::INFINITY];
    let b = vec![1.0f32, 0.0f32]; // 0 * Inf is the dangerous case

    let out = device
        .gemm_f32(1, 2, 1, &a, &b)
        .expect("GEMM should succeed with Inf input");
    assert_eq!(out.len(), 1);
    assert!(
        out[0].is_finite(),
        "GEMM 0*Inf case produced non-finite output: {}",
        out[0]
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_crown_per_position_parallel_gpu_accelerated() {
    use ndarray::Array2;
    use ny_propagate::layers::{GELULayer, LinearLayer};
    use ny_propagate::{GraphNode, Layer};

    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Build a small MLP graph: Linear -> GELU -> Linear
    let in_features = 4;
    let hidden_features = 8;
    let out_features = 4;

    let weight1 = Array2::from_shape_fn((hidden_features, in_features), |(i, j)| {
        0.1 * ((i + j) as f32 - 6.0)
    });
    let bias1 = ndarray::Array1::from_elem(hidden_features, 0.05_f32);
    let linear1 = LinearLayer::new(weight1, Some(bias1)).unwrap();

    let weight2 = Array2::from_shape_fn((out_features, hidden_features), |(i, j)| {
        0.05 * ((i + j) as f32 - 4.0)
    });
    let bias2 = ndarray::Array1::from_elem(out_features, 0.02_f32);
    let linear2 = LinearLayer::new(weight2, Some(bias2)).unwrap();

    let gelu = GELULayer::default();

    let mut graph = ny_propagate::GraphNetwork::new();
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

    let input_lower = Array2::from_elem((1, in_features), -1.0f32);
    let input_upper = Array2::from_elem((1, in_features), 1.0f32);
    let input =
        ny_tensor::BoundedTensor::new(input_lower.into_dyn(), input_upper.into_dyn()).unwrap();

    let start = Instant::now();
    let output = device.crown_per_position_parallel(&graph, &input).unwrap();
    let _duration = start.elapsed();

    // Verify output shape
    let out_shape = output.shape();
    assert_eq!(out_shape, vec![1, out_features]);
    assert_bounds_valid(&output);

    // GPU CROWN must numerically agree with CPU CROWN for this graph/input.
    let cpu_output = graph.propagate_crown_per_position(&input).unwrap();
    assert_bounded_tensor_close(
        &output,
        &cpu_output,
        GPU_REGRESSION_RELAXED_EPSILON,
        "crown_per_position_parallel_gpu_vs_cpu",
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_softmax_ibp_small() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let input_lower = ndarray::array![[0.0, -1.0, 2.0]].into_dyn();
    let input_upper = ndarray::array![[0.5, 0.0, 2.5]].into_dyn();
    let input = ny_tensor::BoundedTensor::new(input_lower, input_upper).unwrap();

    let output = device.softmax_ibp(&input).unwrap();
    let shape = output.shape();

    // Softmax output should have same shape
    assert_eq!(shape, vec![1, 3]);

    assert_bounds_valid(&output);

    // Softmax bounds should sit in [0, 1]
    let lower_bounds = output.lower();
    let upper_bounds = output.upper();
    let eps = 1e-5_f32;
    assert_eq!(lower_bounds.len(), upper_bounds.len());
    for (idx, (&lower, &upper)) in lower_bounds.iter().zip(upper_bounds.iter()).enumerate() {
        assert!(lower >= -eps, "softmax lower[{idx}] out of range: {lower}");
        assert!(
            upper <= 1.0 + eps,
            "softmax upper[{idx}] out of range: {upper}"
        );
    }
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_softmax_ibp_nan_inputs() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Deliberately non-finite bounds to exercise shader NaN/Inf guards.
    let input_lower = ndarray::array![[f32::NAN, -2.0f32, f32::NEG_INFINITY, 0.0f32]].into_dyn();
    let input_upper = ndarray::array![[f32::INFINITY, 2.0f32, f32::NAN, f32::INFINITY]].into_dyn();
    let input = ny_tensor::BoundedTensor::new_unchecked(input_lower, input_upper)
        .expect("new_unchecked accepts non-finite bounds for NaN guard testing");

    let output = device
        .softmax_ibp(&input)
        .expect("softmax_ibp should sanitize NaN/Inf inputs");
    assert_eq!(output.shape(), vec![1, 4]);
    assert_bounds_valid(&output);

    // Softmax bounds are probabilities; all entries must remain in [0, 1].
    let eps = 1e-6_f32;
    for (idx, (&l, &u)) in output.lower().iter().zip(output.upper().iter()).enumerate() {
        assert!(l >= -eps, "softmax lower[{idx}] out of range: {l}");
        assert!(u <= 1.0 + eps, "softmax upper[{idx}] out of range: {u}");
    }
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_scale_ibp_nan_inputs() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // scale=0 with Inf bounds forces 0*Inf => NaN, which must sanitize to fallback bounds.
    let input_lower = vec![f32::INFINITY, -1.0f32, 0.0f32];
    let input_upper = vec![f32::INFINITY, 1.0f32, f32::INFINITY];
    let (out_lower, out_upper) =
        run_scale_ibp_shader_for_test(&device, &input_lower, &input_upper, 0.0);

    for (idx, (&l, &u)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(l.is_finite(), "scale output lower[{idx}] not finite: {l}");
        assert!(u.is_finite(), "scale output upper[{idx}] not finite: {u}");
        assert!(
            l <= u,
            "invalid scale bounds at {idx}: lower={l} > upper={u}"
        );
    }

    // 0*Inf must sanitize conservatively to full fallback range.
    assert!(
        out_lower[0] <= -0.99 * FALLBACK_BOUND,
        "expected lower fallback for 0*Inf, got {}",
        out_lower[0]
    );
    assert!(
        out_upper[0] >= 0.99 * FALLBACK_BOUND,
        "expected upper fallback for 0*Inf, got {}",
        out_upper[0]
    );

    // Finite inputs with scale=0 should remain near zero.
    assert!(
        out_lower[1].abs() <= 1e-6 && out_upper[1].abs() <= 1e-6,
        "finite interval with scale=0 should map near zero, got [{}, {}]",
        out_lower[1],
        out_upper[1]
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_attention_ibp_small() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Small attention: batch=1, heads=2, seq=3, dim=4
    let shape = (1, 2, 3, 4);
    let size = shape.0 * shape.1 * shape.2 * shape.3;

    // Q, K, V bounds
    let q_lower = ndarray::Array::from_elem(shape, -0.5f32).into_dyn();
    let q_upper = ndarray::Array::from_elem(shape, 0.5f32).into_dyn();
    let k_lower = ndarray::Array::from_elem(shape, -0.3f32).into_dyn();
    let k_upper = ndarray::Array::from_elem(shape, 0.3f32).into_dyn();
    let v_lower = ndarray::Array::from_elem(shape, -0.2f32).into_dyn();
    let v_upper = ndarray::Array::from_elem(shape, 0.2f32).into_dyn();

    let q = ny_tensor::BoundedTensor::new(q_lower, q_upper).unwrap();
    let k = ny_tensor::BoundedTensor::new(k_lower, k_upper).unwrap();
    let v = ny_tensor::BoundedTensor::new(v_lower, v_upper).unwrap();

    let scale = 1.0 / (shape.3 as f32).sqrt();
    let output = device.attention_ibp(&q, &k, &v, scale).unwrap();

    // Output shape should be same as Q: [batch, heads, seq, dim]
    assert_eq!(output.shape(), vec![1, 2, 3, 4]);

    assert_bounds_valid(&output);
    let lower = output.lower();
    let upper = output.upper();
    assert_eq!(lower.len(), size);
    assert_eq!(upper.len(), size);

    // GPU attention bounds should match CPU reference within tolerance.
    let cpu_device = crate::ComputeDevice::new(crate::Backend::Cpu).unwrap();
    let cpu_output = cpu_device.attention_ibp(&q, &k, &v, scale).unwrap();
    assert_bounded_tensor_close(
        &output,
        &cpu_output,
        GPU_REGRESSION_RELAXED_EPSILON,
        "attention_ibp_gpu_vs_cpu",
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_attention_ibp_fused_small() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Small attention: batch=1, heads=2, seq=3, dim=4
    let shape = (1, 2, 3, 4);
    let size = shape.0 * shape.1 * shape.2 * shape.3;

    // Q, K, V bounds
    let q_lower = ndarray::Array::from_elem(shape, -0.5f32).into_dyn();
    let q_upper = ndarray::Array::from_elem(shape, 0.5f32).into_dyn();
    let k_lower = ndarray::Array::from_elem(shape, -0.3f32).into_dyn();
    let k_upper = ndarray::Array::from_elem(shape, 0.3f32).into_dyn();
    let v_lower = ndarray::Array::from_elem(shape, -0.2f32).into_dyn();
    let v_upper = ndarray::Array::from_elem(shape, 0.2f32).into_dyn();

    let q = ny_tensor::BoundedTensor::new(q_lower, q_upper).unwrap();
    let k = ny_tensor::BoundedTensor::new(k_lower, k_upper).unwrap();
    let v = ny_tensor::BoundedTensor::new(v_lower, v_upper).unwrap();

    let scale = 1.0 / (shape.3 as f32).sqrt();
    let output = device.attention_ibp_fused(&q, &k, &v, scale).unwrap();

    // Output shape should be same as Q: [batch, heads, seq, dim]
    assert_eq!(output.shape(), vec![1, 2, 3, 4]);

    assert_bounds_valid(&output);
    let lower = output.lower();
    let upper = output.upper();
    assert_eq!(lower.len(), size);
    assert_eq!(upper.len(), size);

    // Fused GPU attention path must match CPU attention reference bounds.
    let cpu_device = crate::ComputeDevice::new(crate::Backend::Cpu).unwrap();
    let cpu_output = cpu_device.attention_ibp(&q, &k, &v, scale).unwrap();
    assert_bounded_tensor_close(
        &output,
        &cpu_output,
        GPU_REGRESSION_RELAXED_EPSILON,
        "attention_ibp_fused_gpu_vs_cpu",
    );
}

// ==================== sanitize_readback unit tests (no GPU required) ====================

/// Regression test #2785: sanitize_readback replaces NaN/Inf with ±FALLBACK_BOUND.
#[test]
fn test_sanitize_readback_nan_inf_replaced_2785() {
    use crate::FALLBACK_BOUND;

    let mut lower = vec![1.0f32, f32::NAN, f32::NEG_INFINITY, 4.0];
    let mut upper = vec![2.0f32, 3.0, f32::INFINITY, f32::NAN];

    super::sanitize_readback(&mut lower, &mut upper);

    // Index 0: both finite → unchanged
    assert_eq!(lower[0], 1.0);
    assert_eq!(upper[0], 2.0);
    // Index 1: lower is NaN → both replaced
    assert_eq!(lower[1], -FALLBACK_BOUND);
    assert_eq!(upper[1], FALLBACK_BOUND);
    // Index 2: lower -Inf, upper Inf → both replaced
    assert_eq!(lower[2], -FALLBACK_BOUND);
    assert_eq!(upper[2], FALLBACK_BOUND);
    // Index 3: upper is NaN → both replaced
    assert_eq!(lower[3], -FALLBACK_BOUND);
    assert_eq!(upper[3], FALLBACK_BOUND);
}

/// Regression test #2785: sanitize_readback with all-finite data is no-op.
#[test]
fn test_sanitize_readback_finite_unchanged_2785() {
    let mut lower = vec![-5.0f32, 0.0, 100.0];
    let mut upper = vec![5.0f32, 1.0, 200.0];
    let lower_orig = lower.clone();
    let upper_orig = upper.clone();

    super::sanitize_readback(&mut lower, &mut upper);

    assert_eq!(lower, lower_orig);
    assert_eq!(upper, upper_orig);
}

/// Regression test #3307: sanitize_readback must repair finite inverted bounds.
#[test]
fn test_sanitize_readback_finite_inversion_repaired_3307() {
    use crate::FALLBACK_BOUND;

    let mut lower = vec![3.0f32, -2.0];
    let mut upper = vec![1.0f32, 4.0];

    super::sanitize_readback(&mut lower, &mut upper);

    assert_eq!(lower[0], -FALLBACK_BOUND);
    assert_eq!(upper[0], FALLBACK_BOUND);
    assert_eq!(lower[1], -2.0);
    assert_eq!(upper[1], 4.0);
}

/// Parallel regression for #3877: linear_ibp staging buffer race.
///
/// Before the fix, 4 IBP ops (linear_ibp, matmul_ibp, softmax_ibp,
/// attention_ibp_fused) dropped the BufferPool MutexGuard before readback,
/// allowing concurrent Rayon threads to recycle staging buffers while
/// `map_async`/`unmap` was still in flight. This test exercises the
/// shared-device linear_ibp path from multiple Rayon threads concurrently.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_linear_ibp_parallel_shared_device_staging_safety_3877() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let batch_size = 4usize;
    let in_features = 16usize;
    let out_features = 8usize;

    // Deterministic input: lower < upper at every element.
    let input_lower: Vec<f32> = (0..(batch_size * in_features))
        .map(|i| (i % 7) as f32 * -0.1 - 0.5)
        .collect();
    let input_upper: Vec<f32> = input_lower.iter().map(|&v| v + 1.0).collect();
    let input_lower_nd =
        ndarray::ArrayD::from_shape_vec(vec![batch_size, in_features], input_lower).unwrap();
    let input_upper_nd =
        ndarray::ArrayD::from_shape_vec(vec![batch_size, in_features], input_upper).unwrap();
    let input =
        ny_tensor::BoundedTensor::new(input_lower_nd.clone(), input_upper_nd.clone()).unwrap();

    // Weight matrix with mixed signs to exercise both wp and wn paths.
    let weight_data: Vec<f32> = (0..(in_features * out_features))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.15)
        .collect();
    // Weight convention: (out_features, in_features) — ncols=in_features, nrows=out_features.
    let weight = ndarray::Array2::from_shape_vec((out_features, in_features), weight_data).unwrap();

    // Compute a serial reference result.
    let reference = device.linear_ibp(&input, &weight, None).unwrap();
    assert_bounds_valid(&reference);

    // Run 8 concurrent Rayon threads, each calling linear_ibp 4 times.
    // Before the fix, this could trigger:
    //   "Queue::submit -> Buffer with '...' label is still mapped"
    let results: Vec<ny_tensor::BoundedTensor> = (0..8)
        .into_par_iter()
        .map(|_| {
            let inp = ny_tensor::BoundedTensor::new(input_lower_nd.clone(), input_upper_nd.clone())
                .unwrap();
            let mut last = None;
            for _ in 0..4 {
                last = Some(
                    device
                        .linear_ibp(&inp, &weight, None)
                        .expect("parallel linear_ibp should not hit staging-buffer race (#3877)"),
                );
            }
            last.unwrap()
        })
        .collect();

    // Every parallel result must match the serial reference.
    let ref_lower = reference.lower();
    let ref_upper = reference.upper();
    for (idx, out) in results.iter().enumerate() {
        assert_bounds_valid(out);
        let out_lower = out.lower();
        let out_upper = out.upper();
        assert_eq!(
            out_lower.shape(),
            ref_lower.shape(),
            "thread {idx} shape mismatch"
        );
        for (a, e) in out_lower.iter().zip(ref_lower.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-4);
        }
        for (a, e) in out_upper.iter().zip(ref_upper.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-4);
        }
    }
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_resident_ibp_forward_matches_sequential_dense_chain_4081() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            ndarray::arr2(&[[1.0_f32, 0.5], [-0.25, 1.5], [0.75, -1.0]]),
            Some(ndarray::arr1(&[0.1, -0.2, 0.05])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            ndarray::arr2(&[[1.0_f32, -0.5, 0.25], [0.0, 0.75, 1.0]]),
            Some(ndarray::arr1(&[0.0, 0.15])),
        )
        .unwrap(),
    ));

    let input = ny_tensor::BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4, 2]),
            vec![-1.0, -0.75, -0.25, 0.0, 0.5, 0.75, 1.0, 1.25],
        )
        .unwrap(),
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4, 2]),
            vec![-0.5, -0.25, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75],
        )
        .unwrap(),
    )
    .unwrap();

    let expected = network
        .propagate_ibp_with_engine(&input, None)
        .expect("CPU sequential IBP should succeed");

    let layers = vec![
        GpuIbpLayer::Linear {
            weight: Arc::from(vec![1.0_f32, 0.5, -0.25, 1.5, 0.75, -1.0]),
            bias: Some(Arc::from(vec![0.1_f32, -0.2, 0.05])),
            out_features: 3,
            in_features: 2,
        },
        GpuIbpLayer::ReLU { num_elements: 12 },
        GpuIbpLayer::Linear {
            weight: Arc::from(vec![1.0_f32, -0.5, 0.25, 0.0, 0.75, 1.0]),
            bias: Some(Arc::from(vec![0.0_f32, 0.15])),
            out_features: 2,
            in_features: 3,
        },
    ];

    let result = device
        .ibp_forward_gpu(
            &layers,
            input.lower().as_slice().unwrap(),
            input.upper().as_slice().unwrap(),
            input.shape(),
        )
        .expect("resident GPU IBP forward should succeed");

    let actual = ny_tensor::BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&result.output_shape), result.lower_bounds)
            .unwrap(),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&result.output_shape), result.upper_bounds)
            .unwrap(),
    )
    .unwrap();

    assert_bounds_valid(&actual);
    assert_bounded_tensor_close(
        &actual,
        &expected,
        GPU_REGRESSION_RELAXED_EPSILON,
        "resident ibp forward parity",
    );
}

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_resident_ibp_forward_matches_sequential_conv2d_chain_4275() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let mut network = Network::new();
    let conv_kernel = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 1, 2, 2]),
        vec![1.0_f32, -0.5, 0.25, 0.75, -0.25, 0.5, 1.0, -1.0],
    )
    .unwrap();
    let conv_bias = ndarray::arr1(&[0.1_f32, -0.2]);
    network.add_layer(Layer::Conv2d(
        Conv2dLayer::with_input_shape(conv_kernel, Some(conv_bias), (1, 1), (0, 0), 4, 4).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(1)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            ndarray::arr2(&[
                [
                    0.2_f32, -0.1, 0.0, 0.3, 0.5, -0.4, 0.1, -0.2, 0.6, -0.3, 0.2, 0.4, -0.5, 0.7,
                    -0.1, 0.2, 0.3, -0.6,
                ],
                [
                    -0.3_f32, 0.4, 0.1, -0.2, 0.0, 0.5, -0.1, 0.2, -0.4, 0.6, -0.5, 0.3, 0.2, -0.1,
                    0.4, -0.2, 0.1, 0.5,
                ],
                [
                    0.1_f32, 0.2, -0.3, 0.4, -0.5, 0.6, 0.2, -0.1, 0.3, -0.4, 0.5, -0.2, 0.1, 0.2,
                    -0.3, 0.4, -0.5, 0.6,
                ],
            ]),
            Some(ndarray::arr1(&[0.05_f32, -0.1, 0.2])),
        )
        .unwrap(),
    ));

    let input = ny_tensor::BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 1, 4, 4]),
            vec![
                -1.0_f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, -0.5, -0.25,
                0.25, 0.5, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, -0.4,
                -0.2, 0.2, 0.4,
            ],
        )
        .unwrap(),
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 1, 4, 4]),
            vec![
                -0.5_f32, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 0.0, 0.25, 0.75,
                1.0, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 0.1, 0.3, 0.7,
                0.9,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let expected = network
        .propagate_ibp_with_engine(&input, None)
        .expect("CPU sequential Conv2d IBP should succeed");

    let layers = vec![
        GpuIbpLayer::Conv2d {
            weight: Arc::from(vec![1.0_f32, -0.5, 0.25, 0.75, -0.25, 0.5, 1.0, -1.0]),
            bias: Some(Arc::from(vec![0.1_f32, -0.2])),
            out_channels: 2,
            in_channels: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            groups: 1,
            input_h: 4,
            input_w: 4,
        },
        GpuIbpLayer::ReLU { num_elements: 36 },
        GpuIbpLayer::View {
            output_shape: Arc::from(vec![2usize, 18usize]),
        },
        GpuIbpLayer::Linear {
            weight: Arc::from(vec![
                0.2_f32, -0.1, 0.0, 0.3, 0.5, -0.4, 0.1, -0.2, 0.6, -0.3, 0.2, 0.4, -0.5, 0.7,
                -0.1, 0.2, 0.3, -0.6, -0.3, 0.4, 0.1, -0.2, 0.0, 0.5, -0.1, 0.2, -0.4, 0.6, -0.5,
                0.3, 0.2, -0.1, 0.4, -0.2, 0.1, 0.5, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.2, -0.1,
                0.3, -0.4, 0.5, -0.2, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6,
            ]),
            bias: Some(Arc::from(vec![0.05_f32, -0.1, 0.2])),
            out_features: 3,
            in_features: 18,
        },
    ];

    let result = device
        .ibp_forward_gpu(
            &layers,
            input.lower().as_slice().unwrap(),
            input.upper().as_slice().unwrap(),
            input.shape(),
        )
        .expect("resident GPU Conv2d IBP forward should succeed");

    let actual = ny_tensor::BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&result.output_shape), result.lower_bounds)
            .unwrap(),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&result.output_shape), result.upper_bounds)
            .unwrap(),
    )
    .unwrap();

    assert_bounds_valid(&actual);
    assert_bounded_tensor_close(
        &actual,
        &expected,
        GPU_REGRESSION_RELAXED_EPSILON,
        "resident conv2d ibp forward parity",
    );
}

// ============================================================================
// DAG resident IBP forward tests (#4319)
// ============================================================================

/// CPU reference for a 2×2 residual DAG: Linear→ReLU→Add(relu, input).
/// Returns (expected_lower, expected_upper) via interval arithmetic.
#[cfg(feature = "gpu-tests")]
fn cpu_residual_dag_ibp(
    weight: &[f32],
    bias: &[f32],
    input_lower: &[f32],
    input_upper: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let w_pos: Vec<f32> = weight.iter().map(|&w| w.max(0.0)).collect();
    let w_neg: Vec<f32> = weight.iter().map(|&w| w.min(0.0)).collect();
    let mut lin_lower = [0.0f32; 2];
    let mut lin_upper = [0.0f32; 2];
    for j in 0..2 {
        for i in 0..2 {
            lin_lower[j] += w_pos[j * 2 + i] * input_lower[i] + w_neg[j * 2 + i] * input_upper[i];
            lin_upper[j] += w_pos[j * 2 + i] * input_upper[i] + w_neg[j * 2 + i] * input_lower[i];
        }
        lin_lower[j] += bias[j];
        lin_upper[j] += bias[j];
    }
    let relu_lower: Vec<f32> = lin_lower.iter().map(|&x| x.max(0.0)).collect();
    let relu_upper: Vec<f32> = lin_upper.iter().map(|&x| x.max(0.0)).collect();
    let exp_l: Vec<f32> = relu_lower
        .iter()
        .zip(input_lower)
        .map(|(&a, &b)| a + b)
        .collect();
    let exp_u: Vec<f32> = relu_upper
        .iter()
        .zip(input_upper)
        .map(|(&a, &b)| a + b)
        .collect();
    (exp_l, exp_u)
}

/// DAG: input(2) → Linear(2→2) → ReLU → Add(relu, input) — residual skip.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_dag_ibp_forward_residual_add_matches_cpu_4319() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let weight: Vec<f32> = vec![0.5, -0.3, 0.2, 0.8];
    let bias: Vec<f32> = vec![0.1, -0.05];
    let input_lower: Vec<f32> = vec![-1.0, 0.5];
    let input_upper: Vec<f32> = vec![1.0, 2.0];

    let plan = GpuDagIbpPlanDesc {
        input_shape: vec![2],
        ops: vec![
            GpuDagIbpOp::Linear {
                weight: Arc::from(weight.clone()),
                bias: Some(Arc::from(bias.clone())),
                out_features: 2,
                in_features: 2,
                input_idx: NETWORK_INPUT_IDX,
            },
            GpuDagIbpOp::ReLU {
                num_elements: 2,
                input_idx: 0,
            },
            GpuDagIbpOp::Add {
                num_elements: 2,
                input_a_idx: 1,
                input_b_idx: NETWORK_INPUT_IDX,
            },
        ],
        output_op_idx: 2,
    };

    let (expected_lower, expected_upper) =
        cpu_residual_dag_ibp(&weight, &bias, &input_lower, &input_upper);

    let cached_plan = device
        .prepare_dag_model_plan(&plan)
        .expect("DAG plan preparation should succeed")
        .expect("DAG plan should not be None for non-empty ops");
    let result = cached_plan
        .dag_ibp_forward_cached(&input_lower, &input_upper, &[2])
        .expect("DAG IBP forward should succeed");

    assert_eq!(result.output_shape, vec![2]);
    for i in 0..2 {
        assert!(result.lower_bounds[i].is_finite() && result.upper_bounds[i].is_finite());
        assert!(result.lower_bounds[i] <= result.upper_bounds[i]);
        assert_relative_eq!(
            result.lower_bounds[i],
            expected_lower[i],
            epsilon = GPU_REGRESSION_RELAXED_EPSILON
        );
        assert_relative_eq!(
            result.upper_bounds[i],
            expected_upper[i],
            epsilon = GPU_REGRESSION_RELAXED_EPSILON
        );
    }
}

/// Test DAG plan with empty ops returns None (fail-closed).
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_dag_ibp_forward_empty_plan_returns_none_4319() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let plan = GpuDagIbpPlanDesc {
        input_shape: vec![4],
        ops: vec![],
        output_op_idx: 0,
    };

    let result = device
        .prepare_dag_model_plan(&plan)
        .expect("empty DAG plan should not error");
    assert!(result.is_none(), "empty DAG plan should return None");
}

/// Test DAG plan rejects shape mismatch on forward.
#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_dag_ibp_forward_shape_mismatch_error_4319() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let plan = GpuDagIbpPlanDesc {
        input_shape: vec![4],
        ops: vec![GpuDagIbpOp::ReLU {
            num_elements: 4,
            input_idx: NETWORK_INPUT_IDX,
        }],
        output_op_idx: 0,
    };

    let cached_plan = device
        .prepare_dag_model_plan(&plan)
        .expect("plan preparation should succeed")
        .expect("plan should not be None");

    // Pass wrong shape — should fail with shape mismatch error
    let result = cached_plan.dag_ibp_forward_cached(
        &[1.0, 2.0, 3.0, 4.0],
        &[2.0, 3.0, 4.0, 5.0],
        &[2, 2], // wrong shape (4 elements but [2,2] != [4])
    );
    assert!(result.is_err(), "should reject mismatched input shape");
}

// ============================================================================
// SOUND graph-DAG resident IBP forward oracle tests (T1.0)
//
// Each sound DAG op emits `[low − r_lo, high + r_hi]` where the directed radius
// over-bounds every f32 rounding error, so by induction over topological order the
// GPU box ENCLOSES both the true forward range and any tighter (fast/exact)
// interval. These oracles prove containment on the live execution path plus the
// routing/degrade contract. Subnormal behavior is characterized by the adapter
// probe, while the FTZ-flush term is checked independently by construction.
// ============================================================================

/// Concrete f64 forward of the residual DAG: `out = relu(W·x + b) + x`.
#[cfg(feature = "gpu-tests")]
fn concrete_residual(weight: &[f32], bias: &[f32], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0f64; n];
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc = f64::from(bias[j]);
        for (i, &xi) in x.iter().enumerate() {
            acc += f64::from(weight[j * n + i]) * xi;
        }
        *o = acc.max(0.0) + x[j];
    }
    out
}

/// Deterministic LCG returning a value in `[0, 1)` — reproducible MC sampling with
/// no `rand` dependency (the workflow flagged that a finite `FALLBACK_BOUND=1e10`
/// must not mask an unsound bound, so the samples must be non-saturating).
#[cfg(feature = "gpu-tests")]
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// input(2) → Linear(2→2) → ReLU → Add(relu, input): the SOUND plan must (1) be a
/// superset of the FAST plan on the same topology (outward widening only), and (2)
/// enclose the true concrete forward at every MC-sampled point (real soundness).
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_dag_ibp_residual_encloses_concrete_and_fast_t1_0() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let weight: Vec<f32> = vec![0.5, -0.3, 0.2, 0.8];
    let bias: Vec<f32> = vec![0.1, -0.05];
    let input_lower: Vec<f32> = vec![-1.0, 0.5];
    let input_upper: Vec<f32> = vec![1.0, 2.0];

    let plan = GpuDagIbpPlanDesc {
        input_shape: vec![2],
        ops: vec![
            GpuDagIbpOp::Linear {
                weight: Arc::from(weight.clone()),
                bias: Some(Arc::from(bias.clone())),
                out_features: 2,
                in_features: 2,
                input_idx: NETWORK_INPUT_IDX,
            },
            GpuDagIbpOp::ReLU {
                num_elements: 2,
                input_idx: 0,
            },
            GpuDagIbpOp::Add {
                num_elements: 2,
                input_a_idx: 1,
                input_b_idx: NETWORK_INPUT_IDX,
            },
        ],
        output_op_idx: 2,
    };

    assert!(
        !device.provides_sound_gpu_dag_ibp(),
        "candidate DAG enclosure tests must not lift the WGPU verdict quarantine"
    );

    let sound_plan = device
        .prepare_sound_dag_model_plan(&plan)
        .expect("sound DAG plan prep should succeed")
        .expect("non-empty ops → Some plan");
    let sound = sound_plan
        .dag_ibp_forward_cached(&input_lower, &input_upper, &[2])
        .expect("sound DAG forward should succeed");
    assert_eq!(sound.output_shape, vec![2]);

    // FAST plan on the same DAG (already validated ≈ exact interval elsewhere).
    let fast_plan = device
        .prepare_dag_model_plan(&plan)
        .expect("fast DAG plan prep")
        .expect("non-empty ops → Some plan");
    let fast = fast_plan
        .dag_ibp_forward_cached(&input_lower, &input_upper, &[2])
        .expect("fast DAG forward");

    // (1) sound ⊇ fast — the sound plan only ever widens outward.
    for i in 0..2 {
        assert!(
            sound.lower_bounds[i] <= fast.lower_bounds[i] + 1e-6,
            "sound lower {} must be <= fast lower {}",
            sound.lower_bounds[i],
            fast.lower_bounds[i]
        );
        assert!(
            sound.upper_bounds[i] >= fast.upper_bounds[i] - 1e-6,
            "sound upper {} must be >= fast upper {}",
            sound.upper_bounds[i],
            fast.upper_bounds[i]
        );
        assert!(sound.lower_bounds[i] <= sound.upper_bounds[i]);
        assert!(sound.lower_bounds[i].is_finite() && sound.upper_bounds[i].is_finite());
    }

    // (2) sound ⊇ concrete forward at 4000 MC-sampled points (STRICT — no slack: an
    // unsound bound must fail here).
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..4000 {
        let x: Vec<f64> = (0..2)
            .map(|i| {
                let t = lcg_unit(&mut state);
                f64::from(input_lower[i]) + t * f64::from(input_upper[i] - input_lower[i])
            })
            .collect();
        let y = concrete_residual(&weight, &bias, &x);
        for i in 0..2 {
            assert!(
                f64::from(sound.lower_bounds[i]) <= y[i],
                "UNSOUND: sound lower {} > concrete {}",
                sound.lower_bounds[i],
                y[i]
            );
            assert!(
                f64::from(sound.upper_bounds[i]) >= y[i],
                "UNSOUND: sound upper {} < concrete {}",
                sound.upper_bounds[i],
                y[i]
            );
        }
    }
}

/// Conv2d(1→2,3×3,pad1) → ReLU → AvgPool(2×2) → View([8]) → Linear(8→3): the sound
/// plan over a MIXED reduction/pool/reshape DAG must enclose the fast plan — the
/// composition cross-check for sound Conv2d + AvgPool + View + Linear.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_dag_ibp_mini_cnn_encloses_fast_t1_0() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Conv weight [out_c=2, in_c=1, 3, 3], mixed signs.
    let conv_w: Vec<f32> = (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let conv_b: Vec<f32> = vec![0.05, -0.1];
    let lin_w: Vec<f32> = (0..24).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let lin_b: Vec<f32> = vec![0.0, 0.2, -0.15];

    let plan = GpuDagIbpPlanDesc {
        input_shape: vec![1, 4, 4],
        ops: vec![
            GpuDagIbpOp::Conv2d {
                weight: Arc::from(conv_w),
                bias: Some(Arc::from(conv_b)),
                out_channels: 2,
                in_channels: 1,
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
                groups: 1,
                input_h: 4,
                input_w: 4,
                input_idx: NETWORK_INPUT_IDX,
            },
            GpuDagIbpOp::ReLU {
                num_elements: 32,
                input_idx: 0,
            },
            GpuDagIbpOp::AveragePool {
                channels: 2,
                input_h: 4,
                input_w: 4,
                output_h: 2,
                output_w: 2,
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
                pad_h: 0,
                pad_w: 0,
                count_include_pad: false,
                is_global: false,
                num_elements: 8,
                input_idx: 1,
            },
            GpuDagIbpOp::View {
                output_shape: Arc::from(vec![8]),
                input_idx: 2,
            },
            GpuDagIbpOp::Linear {
                weight: Arc::from(lin_w),
                bias: Some(Arc::from(lin_b)),
                out_features: 3,
                in_features: 8,
                input_idx: 3,
            },
        ],
        output_op_idx: 4,
    };

    let n_in = 16usize;
    let input_lower: Vec<f32> = (0..n_in).map(|i| -0.5 - (i % 3) as f32 * 0.1).collect();
    let input_upper: Vec<f32> = (0..n_in).map(|i| 0.5 + (i % 4) as f32 * 0.1).collect();

    let sound_plan = device
        .prepare_sound_dag_model_plan(&plan)
        .expect("sound mini-CNN plan prep")
        .expect("non-empty → Some");
    let sound = sound_plan
        .dag_ibp_forward_cached(&input_lower, &input_upper, &[1, 4, 4])
        .expect("sound mini-CNN forward");
    assert_eq!(sound.output_shape, vec![3]);

    let fast_plan = device
        .prepare_dag_model_plan(&plan)
        .expect("fast mini-CNN plan prep")
        .expect("non-empty → Some");
    let fast = fast_plan
        .dag_ibp_forward_cached(&input_lower, &input_upper, &[1, 4, 4])
        .expect("fast mini-CNN forward");

    for i in 0..3 {
        assert!(
            sound.lower_bounds[i] <= fast.lower_bounds[i] + 1e-5,
            "sound lower {} must enclose fast lower {}",
            sound.lower_bounds[i],
            fast.lower_bounds[i]
        );
        assert!(
            sound.upper_bounds[i] >= fast.upper_bounds[i] - 1e-5,
            "sound upper {} must enclose fast upper {}",
            sound.upper_bounds[i],
            fast.upper_bounds[i]
        );
        assert!(sound.lower_bounds[i] <= sound.upper_bounds[i]);
        assert!(sound.lower_bounds[i].is_finite() && sound.upper_bounds[i].is_finite());
    }
}

/// Quarantine + candidate degrade contract: the verdict capability is false,
/// an empty diagnostic plan is `Ok(None)`, and grouped Conv2d is rejected.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_dag_ibp_degrade_and_routing_t1_0() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    assert!(!device.provides_sound_gpu_dag_ibp());

    let empty = GpuDagIbpPlanDesc {
        input_shape: vec![4],
        ops: vec![],
        output_op_idx: 0,
    };
    assert!(
        device
            .prepare_sound_dag_model_plan(&empty)
            .expect("empty plan is not an error")
            .is_none(),
        "empty ops → None (fail-closed)"
    );

    // Grouped Conv2d (groups=2) is not certified for the sound path → Err.
    let grouped = GpuDagIbpPlanDesc {
        input_shape: vec![4, 4, 4],
        ops: vec![GpuDagIbpOp::Conv2d {
            weight: Arc::from(vec![0.1f32; 4 * 2 * 3 * 3]),
            bias: None,
            out_channels: 4,
            in_channels: 4,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            groups: 2,
            input_h: 4,
            input_w: 4,
            input_idx: NETWORK_INPUT_IDX,
        }],
        output_op_idx: 0,
    };
    assert!(
        device.prepare_sound_dag_model_plan(&grouped).is_err(),
        "grouped Conv2d must be rejected (CPU sound fallback)"
    );
}

/// T1.2 SOUND MaxPool2d CROWN backward: an arbitrary linear frontier on the maxpool
/// OUTPUT, transposed to the INPUT via the winner/i* relaxation, then concretized
/// against the input box, must ENCLOSE the concrete frontier value at every sampled
/// input (3000 MC points, STRICT up to f32 noise). Overlapping windows are exercised
/// by a stride-1 3×3 pool; a stride-2 2×2 pool exercises the disjoint case.
#[test]
#[cfg(feature = "gpu-tests")]
fn maxpool_crown_backward_gpu_sound_encloses_concrete_t1_2() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    for &(kh, kw, sh, sw, ph, pw) in &[(2usize, 2, 2, 2, 0, 0), (3, 3, 1, 1, 1, 1)] {
        let (channels, in_h, in_w) = (2usize, 4usize, 4usize);
        let out_h = (in_h + 2 * ph - kh) / sh + 1;
        let out_w = (in_w + 2 * pw - kw) / sw + 1;
        let input_size = channels * in_h * in_w;
        let output_size = channels * out_h * out_w;
        let num_outputs = 3usize;

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let pre_center: Vec<f32> = (0..input_size).map(|_| rng()).collect();
        let pre_rad: Vec<f32> = (0..input_size)
            .map(|_| (rng() * 0.4).abs() + 0.05)
            .collect();
        let pre_lower: Vec<f32> = (0..input_size)
            .map(|i| pre_center[i] - pre_rad[i])
            .collect();
        let pre_upper: Vec<f32> = (0..input_size)
            .map(|i| pre_center[i] + pre_rad[i])
            .collect();
        let lower_a: Vec<f32> = (0..num_outputs * output_size).map(|_| rng()).collect();
        let upper_a: Vec<f32> = (0..num_outputs * output_size).map(|_| rng()).collect();
        let lower_b: Vec<f32> = (0..num_outputs).map(|_| rng() * 0.3).collect();
        let upper_b: Vec<f32> = (0..num_outputs).map(|_| rng() * 0.3).collect();

        let r = device
            .maxpool_crown_backward_gpu_sound(
                &lower_a,
                &upper_a,
                &lower_b,
                &upper_b,
                &pre_lower,
                &pre_upper,
                num_outputs,
                channels,
                in_h,
                in_w,
                out_h,
                out_w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
            )
            .expect("maxpool crown backward");
        assert_eq!(r.lower_a.len(), num_outputs * input_size);

        // Concretize the GPU frontier (coeff sign-selected ∓ err·|x|) + bias.
        let concretize = |a: &[f32], err: &[f32], b: &[f32], lower: bool| -> Vec<f64> {
            let mut out = vec![0.0f64; num_outputs];
            for (o, oo) in out.iter_mut().enumerate() {
                let mut acc = f64::from(b[o]);
                for j in 0..input_size {
                    let coeff = f64::from(a[o * input_size + j]);
                    let (l, u) = (f64::from(pre_lower[j]), f64::from(pre_upper[j]));
                    let pt = if lower {
                        if coeff >= 0.0 {
                            l
                        } else {
                            u
                        }
                    } else if coeff >= 0.0 {
                        u
                    } else {
                        l
                    };
                    acc += coeff * pt;
                    let xabs = l.abs().max(u.abs());
                    let e = f64::from(err[o * input_size + j]) * xabs;
                    if lower {
                        acc -= e
                    } else {
                        acc += e
                    }
                }
                *oo = acc;
            }
            out
        };
        let gpu_lo = concretize(&r.lower_a, &r.lower_a_err, &r.lower_b, true);
        let gpu_hi = concretize(&r.upper_a, &r.upper_a_err, &r.upper_b, false);

        let maxpool_fwd = |x: &[f32]| -> Vec<f32> {
            let mut out = vec![f32::NEG_INFINITY; output_size];
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut m = f32::NEG_INFINITY;
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ih = (oh * sh + kh_i) as isize - ph as isize;
                                let iw = (ow * sw + kw_i) as isize - pw as isize;
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let v =
                                        x[c * (in_h * in_w) + (ih as usize) * in_w + (iw as usize)];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                        }
                        out[c * (out_h * out_w) + oh * out_w + ow] = m;
                    }
                }
            }
            out
        };

        for _ in 0..3000 {
            let x: Vec<f32> = (0..input_size)
                .map(|i| {
                    let t = f32::midpoint(rng(), 1.0).clamp(0.0, 1.0);
                    pre_lower[i] + t * (pre_upper[i] - pre_lower[i])
                })
                .collect();
            let mp = maxpool_fwd(&x);
            for o in 0..num_outputs {
                let mut fl = f64::from(lower_b[o]);
                let mut fu = f64::from(upper_b[o]);
                for k in 0..output_size {
                    fl += f64::from(lower_a[o * output_size + k]) * f64::from(mp[k]);
                    fu += f64::from(upper_a[o * output_size + k]) * f64::from(mp[k]);
                }
                assert!(
                    gpu_lo[o] <= fl + 1e-4,
                    "UNSOUND lower kh={kh} spec {o}: bound {} > frontier {fl}",
                    gpu_lo[o]
                );
                assert!(
                    fu <= gpu_hi[o] + 1e-4,
                    "UNSOUND upper kh={kh} spec {o}: frontier {fu} > bound {}",
                    gpu_hi[o]
                );
            }
        }
    }
}

/// T2.2 Metal-legality CI (no Apple hardware required): every SOUND WGSL shader the
/// verdict paths dispatch (sequential IBP, DAG IBP, and the standalone graph ops)
/// must parse, validate under Metal's EXACT capability set — which EXCLUDES FLOAT64
/// — and emit MSL. A shader that slipped in an f64 op (illegal on Apple GPUs) would
/// fail `validate` here. This is the COMPILE-TIME half of the Metal enclosure; the
/// FTZ/DAZ subnormal *runtime* behavior still needs the per-adapter live probe;
/// backend names do not imply preservation (the observed plain GB10/Vulkan path
/// also flushes).
/// Gated on `wgpu` (default), NOT `gpu-tests`, so it runs in a GPU-less CI —
/// naga's WGSL→MSL translation is pure CPU.
#[cfg(feature = "wgpu")]
#[test]
fn sound_shaders_translate_to_metal_msl_t2_2() {
    use naga::back::msl;
    use naga::valid::{ValidationFlags, Validator};

    let shaders: [(&str, String); 11] = [
        (
            "linear_ibp_sound",
            super::super::shaders::linear_ibp_sound_source(),
        ),
        (
            "maxpool_crown_sound",
            super::super::shaders::maxpool_crown_sound_source(),
        ),
        (
            "relu_ibp_sound",
            super::super::shaders::relu_ibp_sound_source(),
        ),
        (
            "conv2d_ibp_sound",
            super::super::shaders::conv2d_ibp_sound_source(),
        ),
        (
            "matmul_ibp_sound",
            super::super::shaders::matmul_ibp_sound_source(),
        ),
        (
            "avgpool_ibp_sound",
            super::super::shaders::avgpool_ibp_sound_source(),
        ),
        (
            "add_ibp_sound",
            super::super::shaders::add_ibp_sound_source(),
        ),
        (
            "transpose_ibp_sound",
            super::super::shaders::transpose_ibp_sound_source(),
        ),
        (
            "scale_ibp_sound",
            super::super::shaders::scale_ibp_sound_source(),
        ),
        (
            // Resident CROWN A·W error combine — the #gpu-metal-daz weight-amplified
            // floor targets Metal, so its MSL translation matters (self-contained WGSL).
            "crown_aw_error_combine",
            super::super::shaders::CROWN_AW_ERROR_COMBINE_SHADER.to_string(),
        ),
        (
            "crown_strided_gather",
            super::super::shaders::CROWN_STRIDED_GATHER_SHADER.to_string(),
        ),
    ];

    // Metal's exact capability set (no FLOAT64) — the same gate an Apple target
    // applies. Validating against it turns any accidental f64 into a hard failure.
    let metal_caps = msl::supported_capabilities();

    for (name, source) in &shaders {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: WGSL parse failed: {e:?}"));
        let info = Validator::new(ValidationFlags::all(), metal_caps)
            .validate(&module)
            .unwrap_or_else(|e| {
                panic!("{name}: validation under Metal caps failed (illegal f64?): {e:?}")
            });
        // Buffer bindings are assigned by wgpu at runtime; here we only need MSL
        // codegen to complete, so don't error on the absent per-entry-point map.
        let options = msl::Options {
            fake_missing_bindings: true,
            ..Default::default()
        };
        let (msl_src, _) =
            msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
                .unwrap_or_else(|e| panic!("{name}: MSL emission failed: {e:?}"));
        assert!(
            msl_src.contains("kernel"),
            "{name}: emitted MSL is missing a compute kernel entry point"
        );
    }
}

// ---------------------------------------------------------------------------
// conv_transpose_2d GPU-resident plan cache (#perf dispatch wall)
// ---------------------------------------------------------------------------

/// Build a small synthetic per-group conv_transpose_2d problem.
///
/// Returns `(params, weight_col, a_lower, a_upper)` with deterministic, mixed
/// (positive/negative) values so col2im scatter, GEMM, and sign handling are
/// all exercised. Shapes: S=2 specs, OC=3, IC=2, 4x4 grad, 5x5 input, 3x3
/// kernel, stride 1, pad 0 — chosen small enough to readback yet large enough
/// to span several col2im workgroups worth of distinct values.
#[cfg(feature = "gpu-tests")]
fn synthetic_conv_transpose_case() -> (ny_core::ConvTranspose2dParams, Vec<f32>, Vec<f32>, Vec<f32>)
{
    let params = ny_core::ConvTranspose2dParams {
        num_specs: 2,
        out_channels: 3,
        in_channels: 2,
        out_h: 4,
        out_w: 4,
        in_h: 5,
        in_w: 5,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    let spatial = params.out_h * params.out_w;
    let rows = params.num_specs * spatial;
    let kernel_cols = params.in_channels * params.kernel_h * params.kernel_w;
    let weight_len = params.out_channels * kernel_cols;

    let weight_col: Vec<f32> = (0..weight_len)
        .map(|i| {
            let v = (i as f32 * 0.137).sin() * 0.5;
            // ensure mix of signs and a few exact zeros
            if i % 7 == 0 {
                0.0
            } else {
                v
            }
        })
        .collect();

    let a_len = rows * params.out_channels;
    let a_lower: Vec<f32> = (0..a_len).map(|i| (i as f32 * 0.211).cos() - 0.3).collect();
    let a_upper: Vec<f32> = (0..a_len).map(|i| (i as f32 * 0.211).cos() + 0.4).collect();

    (params, weight_col, a_lower, a_upper)
}

/// The cached, GPU-resident fused pair path must be numerically equal to:
///   (a) the existing non-cached GPU `conv_transpose_2d` (two separate calls), and
///   (b) the CPU `NaiveCpuGemmEngine` reference,
/// AND a re-minted weight `Arc` with bit-identical contents must reuse the
/// resident plan (cache stays size 1 — the content-keyed hit the per-call-`Arc`
/// production caller depends on), while DIFFERENT contents at the same geometry
/// — with the previous `Arc`s dropped first, so the allocator may recycle their
/// addresses — must build a fresh plan whose results match the NEW weights.
///
/// This is the validation for the pure-perf, bit-identical plan cache: equal
/// bounds to both references demonstrates soundness preservation, and the
/// cache-length assertions demonstrate genuine resident-buffer reuse vs. no
/// stale-weight hits.
#[cfg(feature = "gpu-tests")]
#[test]
fn test_conv_transpose_plan_cache_pair_matches_uncached_and_cpu_and_reuses() {
    use ny_core::{GemmEngine, NaiveCpuGemmEngine};

    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    // Start from a clean cache so the length assertions are deterministic.
    device
        .clear_conv_transpose_plan_cache()
        .expect("clear conv_transpose plan cache");
    assert_eq!(
        device.conv_transpose_plan_cache_len(),
        0,
        "cache must be empty after clear"
    );

    let (params, weight_col, a_lower, a_upper) = synthetic_conv_transpose_case();

    // --- CPU reference (NaiveCpuGemmEngine) ---
    let cpu = NaiveCpuGemmEngine;
    let cpu_lower = cpu
        .conv_transpose_2d(&a_lower, &weight_col, &params)
        .expect("cpu reference lower");
    let cpu_upper = cpu
        .conv_transpose_2d(&a_upper, &weight_col, &params)
        .expect("cpu reference upper");

    // --- Non-cached GPU path (two separate calls) ---
    let gpu_lower = device
        .conv_transpose_2d(&a_lower, &weight_col, &params)
        .expect("non-cached gpu lower");
    let gpu_upper = device
        .conv_transpose_2d(&a_upper, &weight_col, &params)
        .expect("non-cached gpu upper");

    // --- Cached, GPU-resident fused pair path ---
    let w_arc: Arc<[f32]> = Arc::from(weight_col.clone());
    let (cached_lower, cached_upper) = device
        .conv_transpose_2d_pair_cached(&a_lower, &a_upper, &w_arc, &params)
        .expect("cached pair path");

    // First miss inserts exactly one plan (keyed on the fused 2*S row count).
    assert_eq!(
        device.conv_transpose_plan_cache_len(),
        1,
        "first cached call must build exactly one plan"
    );

    // Lengths.
    let out_elems = params.num_specs * params.in_channels * params.in_h * params.in_w;
    assert_eq!(cached_lower.len(), out_elems);
    assert_eq!(cached_upper.len(), out_elems);

    // (a) cached == non-cached GPU, bit-identical (same shader, same dispatch,
    //     same GEMM reduction axis OC — only buffer reuse + row stacking differ).
    for i in 0..out_elems {
        assert_eq!(
            cached_lower[i], gpu_lower[i],
            "cached lower[{i}] must equal non-cached GPU lower"
        );
        assert_eq!(
            cached_upper[i], gpu_upper[i],
            "cached upper[{i}] must equal non-cached GPU upper"
        );
    }

    // (b) cached == CPU reference within f32 GEMM-reassociation tolerance.
    for i in 0..out_elems {
        assert_relative_eq!(
            cached_lower[i],
            cpu_lower[i],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
        assert_relative_eq!(
            cached_upper[i],
            cpu_upper[i],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }

    // --- Resident reuse: SAME Arc => cache hit, no new plan, identical result ---
    let (cached_lower2, cached_upper2) = device
        .conv_transpose_2d_pair_cached(&a_lower, &a_upper, &w_arc, &params)
        .expect("cached pair path second call (same Arc)");
    assert_eq!(
        device.conv_transpose_plan_cache_len(),
        1,
        "second call with the SAME weight Arc must reuse the resident plan (no re-upload)"
    );
    assert_eq!(
        cached_lower2, cached_lower,
        "resident reuse must produce identical lower bounds"
    );
    assert_eq!(
        cached_upper2, cached_upper,
        "resident reuse must produce identical upper bounds"
    );

    // --- Content hit: DIFFERENT Arc, identical contents => verified reuse ---
    // The production caller mints a fresh Arc per call, so re-minted identical
    // weights are the cache's entire legitimate hit population.
    let w_arc_diff: Arc<[f32]> = Arc::from(weight_col.clone());
    assert_ne!(
        Arc::as_ptr(&w_arc).cast::<f32>() as usize,
        Arc::as_ptr(&w_arc_diff).cast::<f32>() as usize,
        "test precondition: the two weight Arcs are distinct allocations"
    );
    let (cached_lower3, cached_upper3) = device
        .conv_transpose_2d_pair_cached(&a_lower, &a_upper, &w_arc_diff, &params)
        .expect("cached pair path with re-minted identical weights");
    assert_eq!(
        device.conv_transpose_plan_cache_len(),
        1,
        "bit-identical weight contents must reuse the resident plan (content-keyed hit)"
    );
    assert_eq!(cached_lower3, cached_lower);
    assert_eq!(cached_upper3, cached_upper);

    // --- No stale weights: same geometry, DIFFERENT contents => fresh plan ---
    // Drop the caller-side Arcs first (w_arc_diff's allocation is freed and may
    // be recycled; w_arc's lives on inside the plan): the drop-then-realloc
    // pattern the grouped-conv caller produces every iteration, and exactly the
    // collision a pointer-identity key would turn into a hit against the wrong
    // resident weights. Content keying must key the new weights apart.
    drop(w_arc);
    drop(w_arc_diff);
    let weight_col_alt: Vec<f32> = weight_col.iter().map(|w| w * 2.0 + 0.25).collect();
    let gpu_lower_alt = device
        .conv_transpose_2d(&a_lower, &weight_col_alt, &params)
        .expect("non-cached gpu lower (alt weights)");
    let gpu_upper_alt = device
        .conv_transpose_2d(&a_upper, &weight_col_alt, &params)
        .expect("non-cached gpu upper (alt weights)");
    let w_arc_alt: Arc<[f32]> = Arc::from(weight_col_alt);
    let (cached_lower_alt, cached_upper_alt) = device
        .conv_transpose_2d_pair_cached(&a_lower, &a_upper, &w_arc_alt, &params)
        .expect("cached pair path with different weight contents");
    assert_eq!(
        device.conv_transpose_plan_cache_len(),
        2,
        "different weight contents must build a fresh plan (no stale-weight hit)"
    );
    assert_eq!(
        cached_lower_alt, gpu_lower_alt,
        "cached path must compute with the NEW weights, never a stale resident buffer"
    );
    assert_eq!(
        cached_upper_alt, gpu_upper_alt,
        "cached path must compute with the NEW weights, never a stale resident buffer"
    );

    // Clean up shared device state for other serial GPU tests.
    device
        .clear_conv_transpose_plan_cache()
        .expect("clear conv_transpose plan cache after test");
}

// ============================================================================
// SOUND GPU IBP forward oracle (docs/SOUND_GPU_IBP_PLAN.md §7, T1.1 keystone).
// ============================================================================

/// Build a `(Network, Vec<GpuIbpLayer>)` pair for a flat Linear/ReLU MLP from a
/// list of layer dims. ReLU is inserted after every Linear EXCEPT the last one
/// (standard MLP). `weights[l]` is row-major `(dims[l+1] × dims[l])`; `biases[l]`
/// has length `dims[l+1]`. `batch` is the leading (row) count the flat GPU chain
/// runs — the ReLU element count is `batch × width`.
#[cfg(feature = "gpu-tests")]
fn build_mlp_pair(
    dims: &[usize],
    weights: &[Vec<f32>],
    biases: &[Vec<f32>],
    batch: usize,
) -> (Network, Vec<GpuIbpLayer>) {
    let n_lin = dims.len() - 1;
    let mut network = Network::new();
    let mut gpu_layers: Vec<GpuIbpLayer> = Vec::new();
    for l in 0..n_lin {
        let (ni, no) = (dims[l], dims[l + 1]);
        let wmat =
            ndarray::Array2::from_shape_vec((no, ni), weights[l].clone()).expect("weight shape");
        let bvec = ndarray::Array1::from_vec(biases[l].clone());
        network.add_layer(Layer::Linear(
            LinearLayer::new(wmat, Some(bvec)).expect("linear layer"),
        ));
        gpu_layers.push(GpuIbpLayer::Linear {
            weight: Arc::from(weights[l].clone()),
            bias: Some(Arc::from(biases[l].clone())),
            out_features: no,
            in_features: ni,
        });
        if l < n_lin - 1 {
            network.add_layer(Layer::ReLU(ReLULayer));
            gpu_layers.push(GpuIbpLayer::ReLU {
                num_elements: batch * no,
            });
        }
    }
    (network, gpu_layers)
}

/// Concrete f32 forward of one `[batch, dims[0]]` point through the MLP (Linear
/// then ReLU except last). Returns `[batch, dims_last]` flat. This is a TRUE
/// (point) network evaluation the sound intervals must enclose (S1).
#[cfg(feature = "gpu-tests")]
fn concrete_forward_mlp(
    dims: &[usize],
    weights: &[Vec<f32>],
    biases: &[Vec<f32>],
    batch: usize,
    x: &[f32],
) -> Vec<f32> {
    let n_lin = dims.len() - 1;
    let mut cur = x.to_vec(); // [batch, dims[0]]
    let mut cur_w = dims[0];
    for l in 0..n_lin {
        let (ni, no) = (dims[l], dims[l + 1]);
        assert_eq!(cur_w, ni);
        let mut next = vec![0.0f32; batch * no];
        for b in 0..batch {
            for o in 0..no {
                let mut acc = biases[l][o];
                for i in 0..ni {
                    acc += weights[l][o * ni + i] * cur[b * ni + i];
                }
                next[b * no + o] = acc;
            }
        }
        if l < n_lin - 1 {
            for v in next.iter_mut() {
                *v = v.max(0.0);
            }
        }
        cur = next;
        cur_w = no;
    }
    cur
}

/// S1 + S2 enclosure oracle for the SOUND GPU IBP Linear/ReLU dense chain.
///
/// For randomized MLPs + boxes (incl. adversarial cancellation, large weights,
/// subnormal-range inputs) asserts, elementwise:
///   GPU_sound_lo <= CPU_propagate_ibp_sound_lo         (S2: never tighter than CPU)
///   GPU_sound_hi >= CPU_propagate_ibp_sound_hi
///   GPU_sound_lo <= concrete_sample <= GPU_sound_hi    (S1: encloses TRUE outputs)
/// with ZERO enclosure violations.
///
/// Subnormal note (spec §7/A10): preservation is qualified by the live adapter
/// probe, not inferred from Vulkan. This oracle validates S1/S2 on the execution
/// path it actually receives; the `flush` amplifier is checked by construction
/// in `sound_gpu_ibp_flush_radius_amplified_by_weight_t1_1`.
///
/// The CPU reference is the exact path the GPU replaces: the N-D
/// `linear.propagate_ibp_sound` (which double-widens by `2·(in+2)` ULPs). We assert
/// CPU-encloses-concrete ONLY on the MILD configs; under HEAVY cancellation the CPU
/// N-D n-ULP widen scales with |result| (not Σ|terms|) and can be loose, so there we
/// assert the GPU (which carries the full `γ·S` term) as the sound authority — the
/// task's requirement — and still assert GPU ⊇ CPU.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_linear_encloses_cpu_sound_and_samples_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let gpu: &dyn GpuIbpForward = &*device;

    let mut state: u64 = 0x00A1_CE5E_ED1B_F00D;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    // (dims, weight_scale, box_half_width, cpu_encloses_concrete)
    let configs: &[(&[usize], f32, f32, bool)] = &[
        (&[3, 4, 3], 0.8, 0.3, true),    // small MLP, mild
        (&[5, 6, 4], 0.7, 0.25, true),   // 1 hidden, mild
        (&[4, 8, 6, 3], 0.6, 0.2, true), // 2 hidden, mild
        (&[6, 5], 1.0, 0.4, true),       // single linear, mild
        (&[3, 16], 1.3, 0.35, false),    // wide, heavy cancellation
        (&[4, 4, 4], 1.5, 0.15, false),  // cancellation across depth
        (&[2, 3], 64.0, 0.3, false),     // large weights (2^6)
        (&[2, 3], 1024.0, 0.2, false),   // large weights (2^10)
        (&[3, 4], 1.0, 1e-38, false),    // subnormal-range, live-path-qualified
    ];

    let batch = 2usize;
    let mut total_checks = 0usize;
    for &(dims, wscale, box_hw, cpu_encloses) in configs {
        for _trial in 0..6 {
            let n_lin = dims.len() - 1;
            let weights: Vec<Vec<f32>> = (0..n_lin)
                .map(|l| (0..dims[l + 1] * dims[l]).map(|_| rng() * wscale).collect())
                .collect();
            let biases: Vec<Vec<f32>> = (0..n_lin)
                .map(|l| (0..dims[l + 1]).map(|_| rng() * 0.1).collect())
                .collect();

            let (network, gpu_layers) = build_mlp_pair(dims, &weights, &biases, batch);

            // Input box [batch, dims[0]].
            let in0 = dims[0];
            let mut xl = vec![0.0f32; batch * in0];
            let mut xu = vec![0.0f32; batch * in0];
            for k in 0..batch * in0 {
                let c = rng() * 0.4;
                xl[k] = c - box_hw;
                xu[k] = c + box_hw;
            }
            let input = ny_tensor::BoundedTensor::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch, in0]), xl.clone()).unwrap(),
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch, in0]), xu.clone()).unwrap(),
            )
            .unwrap();

            // GPU sound IBP (the op under test).
            let res = gpu
                .ibp_forward_gpu_sound(&gpu_layers, &xl, &xu, &[batch, in0])
                .expect("sound GPU IBP forward should succeed for a Linear/ReLU chain");
            let gpu_lo = &res.lower_bounds;
            let gpu_hi = &res.upper_bounds;

            // CPU sound reference (the exact path the GPU stands in for).
            let cpu = network.propagate_ibp_sound(&input).expect("CPU sound IBP");
            let cpu_lo: Vec<f32> = cpu.lower().iter().copied().collect();
            let cpu_hi: Vec<f32> = cpu.upper().iter().copied().collect();

            assert_eq!(gpu_lo.len(), cpu_lo.len(), "output element count parity");
            assert_eq!(
                res.output_shape,
                cpu.shape().to_vec(),
                "output shape parity"
            );

            // S2: GPU ⊇ CPU (never tighter), elementwise, zero tolerance.
            for j in 0..gpu_lo.len() {
                assert!(
                    gpu_lo[j].is_finite() && gpu_hi[j].is_finite() && gpu_lo[j] <= gpu_hi[j],
                    "GPU interval invalid at {j}: [{}, {}]",
                    gpu_lo[j],
                    gpu_hi[j]
                );
                assert!(
                    gpu_lo[j] <= cpu_lo[j],
                    "S2 violation dims={dims:?} elem {j}: GPU_lo {} > CPU_lo {}",
                    gpu_lo[j],
                    cpu_lo[j]
                );
                assert!(
                    gpu_hi[j] >= cpu_hi[j],
                    "S2 violation dims={dims:?} elem {j}: GPU_hi {} < CPU_hi {}",
                    gpu_hi[j],
                    cpu_hi[j]
                );
            }

            // S1: GPU (and, for mild configs, CPU) enclose every concrete sample.
            let mut samples: Vec<Vec<f32>> = Vec::new();
            samples.push(xl.clone()); // all-lower corner
            samples.push(xu.clone()); // all-upper corner
            let mid: Vec<f32> = xl.iter().zip(&xu).map(|(l, u)| 0.5 * (l + u)).collect();
            samples.push(mid);
            for _ in 0..6 {
                let s: Vec<f32> = xl
                    .iter()
                    .zip(&xu)
                    .map(|(l, u)| {
                        let t = f32::midpoint(rng(), 1.0); // [0,1]
                        l + t * (u - l)
                    })
                    .collect();
                samples.push(s);
            }

            for s in &samples {
                let y = concrete_forward_mlp(dims, &weights, &biases, batch, s);
                for j in 0..y.len() {
                    if !y[j].is_finite() {
                        continue; // overflow to inf: FALLBACK guards it, skip
                    }
                    assert!(
                        gpu_lo[j] <= y[j] && y[j] <= gpu_hi[j],
                        "S1 (GPU) violation dims={dims:?} elem {j}: y={} not in [{}, {}]",
                        y[j],
                        gpu_lo[j],
                        gpu_hi[j]
                    );
                    if cpu_encloses {
                        assert!(
                            cpu_lo[j] <= y[j] && y[j] <= cpu_hi[j],
                            "CPU reference should enclose concrete on mild config dims={dims:?} elem {j}: y={} not in [{}, {}]",
                            y[j],
                            cpu_lo[j],
                            cpu_hi[j]
                        );
                    }
                    total_checks += 1;
                }
            }
        }
    }
    assert!(total_checks > 0, "oracle must have run enclosure checks");
    eprintln!("sound GPU IBP oracle: {total_checks} enclosure checks, 0 violations");
}

/// Adversarial NaN/inf box → the sound linear degrades to `[-FALLBACK, +FALLBACK]`
/// (a maximal sound superset), never a tight/false interval (spec §7 case (e)).
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_linear_nan_inf_box_degrades_to_fallback_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let gpu: &dyn GpuIbpForward = &*device;

    let layers = vec![GpuIbpLayer::Linear {
        weight: Arc::from(vec![1.0f32, -0.5, 0.25, 2.0]),
        bias: Some(Arc::from(vec![0.0f32, 0.0])),
        out_features: 2,
        in_features: 2,
    }];
    // Row 0: an infinite upper endpoint; row 1: a NaN lower endpoint.
    let xl = vec![0.0f32, 0.0, f32::NAN, 0.0];
    let xu = vec![1.0f32, f32::INFINITY, 1.0, 1.0];
    let res = gpu
        .ibp_forward_gpu_sound(&layers, &xl, &xu, &[2usize, 2usize])
        .expect("sound GPU IBP forward should succeed (degrading, not erroring)");
    for j in 0..res.lower_bounds.len() {
        let (lo, hi) = (res.lower_bounds[j], res.upper_bounds[j]);
        assert!(
            lo.is_finite() && hi.is_finite() && lo <= hi,
            "elem {j}: [{lo}, {hi}]"
        );
        assert!(
            lo >= -FALLBACK_BOUND && hi <= FALLBACK_BOUND,
            "elem {j} must be clamped within ±FALLBACK: [{lo}, {hi}]"
        );
    }
}

/// By-construction check that the sampled execution oracle cannot establish alone
/// (spec §7): the §0 weight-amplified operand-flush floor makes the emitted radius
/// `>= |W|·FLT_MIN` even when the execution path preserves a subnormal input. A
/// weight-INDEPENDENT floor would emit a radius ~90 binary orders of magnitude too
/// tight here (a false-VERIFIED break).
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_flush_radius_amplified_by_weight_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let gpu: &dyn GpuIbpForward = &*device;

    let w = 2.0f32.powi(100); // huge weight
    let x = 2.0f32.powi(-130); // valid subnormal; preservation is execution-path-specific
    let layers = vec![GpuIbpLayer::Linear {
        weight: Arc::from(vec![w]),
        bias: Some(Arc::from(vec![0.0f32])),
        out_features: 1,
        in_features: 1,
    }];
    let res = gpu
        .ibp_forward_gpu_sound(&layers, &[x], &[x], &[1usize, 1usize])
        .expect("sound GPU IBP forward");
    let (lo, hi) = (res.lower_bounds[0], res.upper_bounds[0]);
    let radius = 0.5f32 * (hi - lo);
    // |W|·FLT_MIN = 2^100 · 2^-126 = 2^-26. The §0 floor guarantees radius >= this.
    let flt_min = f32::from_bits(0x0080_0000); // 2^-126, smallest NORMAL
    let amplified_floor = w * flt_min; // 2^-26
    assert!(
        radius >= 0.5 * amplified_floor,
        "sound radius {radius:e} must cover the |W|·FLT_MIN = {amplified_floor:e} amplified-flush \
         floor (§0); lo={lo:e} hi={hi:e}"
    );
    // And the interval still encloses the true product 2^-30 (a NORMAL f32).
    let y = w * x; // 2^-30
    assert!(
        lo <= y && y <= hi,
        "interval [{lo:e}, {hi:e}] must enclose true y = {y:e}"
    );
}

/// A metadata-only `View` (Flatten/Reshape) between dense layers must be a SOUND,
/// EXACT pass-through — NOT an `UnsupportedOp` CPU-fallback (the pre-fix behavior
/// that pushed every flatten-before-FC CNN off the GPU sound IBP). Assert (1) the
/// View chain succeeds, (2) it changes only the shape (here [2,6]→[2,1,6], so the
/// final shape differs from the no-View chain), and (3) the FLAT bounds are
/// bit-identical to the same chain with the no-op reshape removed — a reshape moves
/// no data and adds no rounding, so `GPU_view == GPU_plain` exactly.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_view_is_exact_passthrough_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let gpu: &dyn GpuIbpForward = &*device;

    let w1: Vec<f32> = (0..6 * 4).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
    let w2: Vec<f32> = (0..3 * 6).map(|i| ((i % 5) as f32 - 2.0) * 0.4).collect();
    let lin1 = GpuIbpLayer::Linear {
        weight: Arc::from(w1),
        bias: Some(Arc::from(vec![0.05f32; 6])),
        out_features: 6,
        in_features: 4,
    };
    let lin2 = GpuIbpLayer::Linear {
        weight: Arc::from(w2),
        bias: Some(Arc::from(vec![-0.05f32; 3])),
        out_features: 3,
        in_features: 6,
    };
    // Reshape [2,6] -> [2,1,6]: non-trivial (adds a dim) yet element-preserving and
    // last-dim = next Linear's in_features (6).
    let view = GpuIbpLayer::View {
        output_shape: Arc::from(vec![2usize, 1, 6]),
    };

    let batch = 2usize;
    let xl: Vec<f32> = (0..batch * 4).map(|i| -0.3 - (i as f32) * 0.01).collect();
    let xu: Vec<f32> = (0..batch * 4).map(|i| 0.3 + (i as f32) * 0.01).collect();

    let with_view = vec![lin1.clone(), view, lin2.clone()];
    let without_view = vec![lin1, lin2];

    let r_view = gpu
        .ibp_forward_gpu_sound(&with_view, &xl, &xu, &[batch, 4])
        .expect("View chain must succeed on GPU sound IBP (was UnsupportedOp pre-fix)");
    let r_plain = gpu
        .ibp_forward_gpu_sound(&without_view, &xl, &xu, &[batch, 4])
        .expect("plain chain");

    assert_eq!(
        r_view.output_shape,
        vec![2usize, 1, 3],
        "View must propagate the reshaped leading dims into the output shape"
    );
    assert_eq!(
        r_view.lower_bounds, r_plain.lower_bounds,
        "View must not change the (flat) lower bounds"
    );
    assert_eq!(
        r_view.upper_bounds, r_plain.upper_bounds,
        "View must not change the (flat) upper bounds"
    );
}

// ── Sound GPU IBP graph-op siblings (§3.2–§3.8) — live-path oracles ──────────
//
// These oracles gate the SOUND shader siblings the same way the Linear keystone is
// gated: the emitted f32 interval must be a SUPERSET of the TRUE mathematical range
// of the op over the input box. The true range is computed here in f64 (each product
// of two f32 is exact in f64; a handful summed in f64 carries ~2^-52 error, ~18
// binary orders below the shader's ~2^-24-scale γ widening — so `GPU ⊇ truth`
// admits no false failure). This is S1 (the verdict-critical property). Brute-forced
// concrete corner/interior samples are additionally contained. S2 (`GPU ⊇ CPU
// propagate_ibp_sound`) is asserted for the Conv2d chain path (which IS verdict-
// wired via `ibp_forward_gpu_sound`); the graph-only ops (MatMul/Add/Transpose/
// Scale/AvgPool) are NOT verdict-wired until the DAG accessor forwards the sound
// flag (T1.0), so their oracle asserts the STRONGER `GPU ⊇ exact-f64 truth`.
//
// Subnormal note (spec §7/A10): a Vulkan backend name does not establish
// preservation. The live adapter gate characterizes the arithmetic; the
// `*_flush_*` tests below independently read back the emitted widening.

/// A small deterministic LCG in [-1, 1] for the sound graph-op oracles.
#[cfg(feature = "gpu-tests")]
fn sound_op_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// S1 oracle for the SOUND MatMul IBP shader (§3.3): the emitted interval encloses
/// the exact f64 range of the interval matrix product over the box, including
/// cancellation, large-magnitude, and subnormal-range (live-path-qualified) configs.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_matmul_encloses_truth_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let mut rng = sound_op_rng(0xBEEF_1234_5678_9ABC);
    // (batch, m, k, n, a_scale, b_scale, box_hw)
    let configs: &[(usize, usize, usize, usize, f32, f32, f32)] = &[
        (1, 3, 4, 2, 1.0, 1.0, 0.3),
        (2, 2, 5, 3, 1.0, 1.0, 0.4),
        (1, 4, 8, 4, 2.0, 0.5, 0.2),
        (1, 3, 6, 3, 3.0, 3.0, 0.5),   // heavy cancellation (S ≫ |y|)
        (1, 2, 3, 2, 64.0, 64.0, 0.3), // large magnitudes
        (1, 3, 3, 3, 1.0, 1.0, 1e-38), // subnormal-range, live-path-qualified
    ];
    let mut checks = 0usize;
    for &(batch, m, k, n, ascale, bscale, hw) in configs {
        for _ in 0..4 {
            let (alen, blen) = (batch * m * k, batch * k * n);
            let (mut al, mut au) = (vec![0f32; alen], vec![0f32; alen]);
            let (mut bl, mut bu) = (vec![0f32; blen], vec![0f32; blen]);
            for i in 0..alen {
                let c = rng() * ascale;
                al[i] = c - hw;
                au[i] = c + hw;
            }
            for i in 0..blen {
                let c = rng() * bscale;
                bl[i] = c - hw;
                bu[i] = c + hw;
            }
            let (lo, hi) = device
                .matmul_ibp_sound(&al, &au, &bl, &bu, batch, m, k, n)
                .expect("matmul sound");
            for b in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let mut tlo = 0f64;
                        let mut thi = 0f64;
                        for kk in 0..k {
                            let a_l = f64::from(al[b * m * k + i * k + kk]);
                            let a_u = f64::from(au[b * m * k + i * k + kk]);
                            let b_l = f64::from(bl[b * k * n + kk * n + j]);
                            let b_u = f64::from(bu[b * k * n + kk * n + j]);
                            let (p1, p2, p3, p4) = (a_l * b_l, a_l * b_u, a_u * b_l, a_u * b_u);
                            tlo += p1.min(p2).min(p3.min(p4));
                            thi += p1.max(p2).max(p3.max(p4));
                        }
                        let idx = b * m * n + i * n + j;
                        let (glo, ghi) = (f64::from(lo[idx]), f64::from(hi[idx]));
                        assert!(
                            glo.is_finite() && ghi.is_finite() && glo <= ghi,
                            "matmul interval invalid at {idx}: [{glo}, {ghi}]"
                        );
                        assert!(
                            glo <= tlo,
                            "S1 matmul lo violation b{b} i{i} j{j}: GPU_lo {glo} > truth {tlo}"
                        );
                        assert!(
                            ghi >= thi,
                            "S1 matmul hi violation b{b} i{i} j{j}: GPU_hi {ghi} < truth {thi}"
                        );
                        checks += 1;
                    }
                }
            }
        }
    }
    assert!(checks > 0, "matmul oracle must run checks");
    eprintln!("sound GPU IBP matmul oracle: {checks} enclosure checks, 0 violations");
}

/// S1 oracle for the SOUND element-wise Add IBP shader (§3.5).
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_add_encloses_truth_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let mut rng = sound_op_rng(0xADD0_0F00_1234_5678);
    let configs: &[(usize, f32, f32)] = &[
        (16, 1.0, 0.3),
        (64, 4.0, 0.5),
        (10, 1e30, 0.5),  // near-overflow magnitudes
        (10, 1.0, 1e-38), // subnormal-range box
    ];
    let mut checks = 0usize;
    for &(n, scale, hw) in configs {
        for _ in 0..4 {
            let (mut al, mut au) = (vec![0f32; n], vec![0f32; n]);
            let (mut bl, mut bu) = (vec![0f32; n], vec![0f32; n]);
            for i in 0..n {
                let ca = rng() * scale;
                al[i] = ca - hw;
                au[i] = ca + hw;
                let cb = rng() * scale;
                bl[i] = cb - hw;
                bu[i] = cb + hw;
            }
            let (lo, hi) = device.add_ibp_sound(&al, &au, &bl, &bu).expect("add sound");
            for i in 0..n {
                let tlo = f64::from(al[i]) + f64::from(bl[i]);
                let thi = f64::from(au[i]) + f64::from(bu[i]);
                let (glo, ghi) = (f64::from(lo[i]), f64::from(hi[i]));
                assert!(
                    glo.is_finite() && ghi.is_finite() && glo <= ghi,
                    "add interval invalid at {i}: [{glo}, {ghi}]"
                );
                // Unconditional, including the near-overflow config: a finite
                // endpoint encloses at ANY magnitude, so the shader must pass it
                // through — `FALLBACK_BOUND` is the non-finite sentinel, not a cap
                // (#2549 pins the same contract on the CPU linear path).
                assert!(glo <= tlo, "S1 add lo violation {i}: {glo} > {tlo}");
                assert!(ghi >= thi, "S1 add hi violation {i}: {ghi} < {thi}");
                checks += 1;
            }
        }
    }
    assert!(checks > 0);
    eprintln!("sound GPU IBP add oracle: {checks} enclosure checks, 0 violations");
}

/// #2549 on the GPU: the sound Add passes a FINITE endpoint through at ANY
/// magnitude. `FALLBACK_BOUND` marks a non-finite endpoint whose true value is
/// unknown; a finite sum past it is already a valid enclosure, so clamping it
/// toward zero would emit a box that fails to contain the truth — the
/// false-VERIFIED direction. Both operands here stay inside ±FALLBACK: only the
/// sum crosses it, so no out-of-range input is needed to reach the branch.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_add_preserves_finite_endpoints_past_fallback_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let a = vec![6e9f32, -8e9, 9.9e9];
    let b = vec![6e9f32, -8e9, 9.9e9];
    let (lo, hi) = device.add_ibp_sound(&a, &a, &b, &b).expect("add sound");
    for i in 0..a.len() {
        let t = f64::from(a[i]) + f64::from(b[i]);
        let (glo, ghi) = (f64::from(lo[i]), f64::from(hi[i]));
        assert!(
            t.abs() > f64::from(FALLBACK_BOUND),
            "premise: |{t}| must exceed the sentinel for elem {i} to exercise the branch"
        );
        assert!(
            glo.is_finite() && ghi.is_finite(),
            "add interval invalid at {i}: [{glo}, {ghi}]"
        );
        assert!(glo <= t, "S1 add lo violation {i}: {glo} > {t}");
        assert!(ghi >= t, "S1 add hi violation {i}: {ghi} < {t}");
    }
}

/// S1 oracle for the SOUND Transpose IBP shader (§3.6): the widened permutation
/// still encloses the exact (copied) endpoints.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_transpose_encloses_truth_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let mut rng = sound_op_rng(0x7A05_9057_0011_2233);
    let configs: &[(usize, usize, usize)] = &[(1, 3, 4), (2, 5, 2), (1, 1, 7), (3, 4, 4)];
    let mut checks = 0usize;
    for &(batch, rows, cols) in configs {
        let total = batch * rows * cols;
        let (mut il, mut iu) = (vec![0f32; total], vec![0f32; total]);
        for i in 0..total {
            let c = rng() * 5.0;
            let hw = (rng() + 1.0) * 0.3;
            il[i] = c - hw;
            iu[i] = c + hw;
        }
        let (lo, hi) = device
            .transpose_ibp_sound(&il, &iu, batch, rows, cols)
            .expect("transpose sound");
        // Output [batch, cols, rows]; out[b, oc, orow] = in[b, orow, oc].
        for b in 0..batch {
            for oc in 0..cols {
                for orow in 0..rows {
                    let out_idx = b * (rows * cols) + oc * rows + orow;
                    let in_idx = b * (rows * cols) + orow * cols + oc;
                    let (tlo, thi) = (f64::from(il[in_idx]), f64::from(iu[in_idx]));
                    let (glo, ghi) = (f64::from(lo[out_idx]), f64::from(hi[out_idx]));
                    assert!(
                        glo.is_finite() && ghi.is_finite() && glo <= ghi,
                        "transpose interval invalid: [{glo}, {ghi}]"
                    );
                    assert!(glo <= tlo, "S1 transpose lo violation: {glo} > {tlo}");
                    assert!(ghi >= thi, "S1 transpose hi violation: {ghi} < {thi}");
                    checks += 1;
                }
            }
        }
    }
    assert!(checks > 0);
    eprintln!("sound GPU IBP transpose oracle: {checks} enclosure checks, 0 violations");
}

/// S1 oracle for the SOUND Scale IBP shader (§3.8), including `|s| > 16` (the fixed-
/// floor break) and a live-path-qualified `|s|·subnormal` amplification case.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_scale_encloses_truth_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let mut rng = sound_op_rng(0x5CA1_E000_9988_7766);
    // (n, scale, box_center_scale, box_hw)
    let configs: &[(usize, f32, f32, f32)] = &[
        (16, 1.0, 1.0, 0.3),
        (16, -2.5, 1.0, 0.4),
        (16, 0.0, 5.0, 0.5),         // s == 0 → [0, 0] (Inf·0 guard)
        (16, 1_048_576.0, 1.0, 0.2), // |s| = 2^20 ≫ 16 (fixed-floor break)
        (16, -65536.0, 1.0, 0.3),    // |s| = 2^16, negative
        (16, 3.0, 1.0, 1e-38),       // subnormal-range box
    ];
    let mut checks = 0usize;
    for &(n, s, cscale, hw) in configs {
        for _ in 0..3 {
            let (mut il, mut iu) = (vec![0f32; n], vec![0f32; n]);
            for i in 0..n {
                let c = rng() * cscale;
                il[i] = c - hw;
                iu[i] = c + hw;
            }
            let (lo, hi) = device.scale_ibp_sound(&il, &iu, s).expect("scale sound");
            for i in 0..n {
                let sd = f64::from(s);
                let (a, bb) = (sd * f64::from(il[i]), sd * f64::from(iu[i]));
                let (tlo, thi) = (a.min(bb), a.max(bb));
                let (glo, ghi) = (f64::from(lo[i]), f64::from(hi[i]));
                assert!(
                    glo.is_finite() && ghi.is_finite() && glo <= ghi,
                    "scale interval invalid at {i}: [{glo}, {ghi}]"
                );
                assert!(
                    glo <= tlo,
                    "S1 scale lo violation s={s} i{i}: {glo} > {tlo}"
                );
                assert!(
                    ghi >= thi,
                    "S1 scale hi violation s={s} i{i}: {ghi} < {thi}"
                );
                checks += 1;
            }
        }
    }
    assert!(checks > 0);
    eprintln!("sound GPU IBP scale oracle: {checks} enclosure checks, 0 violations");
}

/// By-construction check that the sampled execution oracle cannot establish alone
/// (spec §7 case d): the §3.8 `|s|`-amplified `scale_floor` makes the emitted
/// half-width `≥ |s|·FLT_MIN` even when the execution path preserves a subnormal
/// input. A fixed `ADDITIVE1` floor (UNSOUND for `|s| > 16`) would be ~14 binary
/// orders too tight here.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_scale_flush_amplified_by_scale_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let s = 2.0f32.powi(20); // |s| = 2^20 ≫ 16
    let largest_subnormal = f32::from_bits(0x007f_ffff); // ≈ 2^-126
                                                         // Box [0, largest_subnormal]: a subnormal upper amplified by |s|.
    let (lo, hi) = device
        .scale_ibp_sound(&[0.0f32], &[largest_subnormal], s)
        .expect("scale sound");
    let (lo, hi) = (lo[0], hi[0]);
    let flt_min = f32::from_bits(0x0080_0000); // 2^-126 smallest NORMAL
    let amplified_floor = f64::from(s) * f64::from(flt_min); // 2^-106
    let half_width = 0.5 * (f64::from(hi) - f64::from(lo));
    assert!(
        half_width >= amplified_floor,
        "scale half-width {half_width:e} must cover |s|·FLT_MIN = {amplified_floor:e} (§3.8)"
    );
    // And the interval encloses the true amplified product s·largest_subnormal.
    let y = f64::from(s) * f64::from(largest_subnormal);
    assert!(
        f64::from(lo) <= y && y <= f64::from(hi),
        "scale interval [{lo:e}, {hi:e}] must enclose true y = {y:e}"
    );
}

/// S1 oracle for the SOUND AvgPool IBP shader (§3.4), windowed and padded configs.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_avgpool_encloses_truth_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let mut rng = sound_op_rng(0xA46_9001_C0FF_EE11);
    // (channels, in_h, in_w, k_h, k_w, s_h, s_w, pad_h, pad_w, count_include_pad)
    let configs: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        bool,
    )] = &[
        (2, 4, 4, 2, 2, 2, 2, 0, 0, false), // clean 2×2 stride-2
        (1, 5, 5, 3, 3, 1, 1, 1, 1, true),  // padded, count_include_pad
        (1, 5, 5, 3, 3, 1, 1, 1, 1, false), // padded, exclude pad
        (3, 6, 6, 6, 6, 1, 1, 0, 0, false), // global pool
    ];
    let mut checks = 0usize;
    for &(ch, ih, iw, kh, kw, sh, sw, ph, pw, cip) in configs {
        let oh = (ih + 2 * ph - kh) / sh + 1;
        let ow = (iw + 2 * pw - kw) / sw + 1;
        let ilen = ch * ih * iw;
        let (mut il, mut iu) = (vec![0f32; ilen], vec![0f32; ilen]);
        for i in 0..ilen {
            let c = rng() * 3.0;
            let hw = (rng() + 1.0) * 0.3;
            il[i] = c - hw;
            iu[i] = c + hw;
        }
        let (lo, hi) = device
            .avgpool_ibp_sound(&il, &iu, ch, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, cip)
            .expect("avgpool sound");
        for c in 0..ch {
            for y in 0..oh {
                for x in 0..ow {
                    let (mut sum_l, mut sum_u) = (0f64, 0f64);
                    let mut count = 0usize;
                    for a in 0..kh {
                        for b in 0..kw {
                            let ihp = (y * sh + a) as isize - ph as isize;
                            let iwp = (x * sw + b) as isize - pw as isize;
                            if ihp >= 0 && (ihp as usize) < ih && iwp >= 0 && (iwp as usize) < iw {
                                let flat = c * ih * iw + (ihp as usize) * iw + (iwp as usize);
                                sum_l += f64::from(il[flat]);
                                sum_u += f64::from(iu[flat]);
                                count += 1;
                            } else if cip {
                                count += 1;
                            }
                        }
                    }
                    let divisor = if cip {
                        (kh * kw) as f64
                    } else {
                        count.max(1) as f64
                    };
                    let (tlo, thi) = (sum_l / divisor, sum_u / divisor);
                    let out_idx = c * oh * ow + y * ow + x;
                    let (glo, ghi) = (f64::from(lo[out_idx]), f64::from(hi[out_idx]));
                    assert!(
                        glo.is_finite() && ghi.is_finite() && glo <= ghi,
                        "avgpool interval invalid: [{glo}, {ghi}]"
                    );
                    assert!(
                        glo <= tlo,
                        "S1 avgpool lo violation c{c} y{y} x{x}: {glo} > {tlo}"
                    );
                    assert!(
                        ghi >= thi,
                        "S1 avgpool hi violation c{c} y{y} x{x}: {ghi} < {thi}"
                    );
                    checks += 1;
                }
            }
        }
    }
    assert!(checks > 0);
    eprintln!("sound GPU IBP avgpool oracle: {checks} enclosure checks, 0 violations");
}

/// S1 + chain-integration oracle for the SOUND Conv2d IBP shader (§3.2), exercised
/// through the verdict-wired `ibp_forward_gpu_sound` dense-chain driver as a single-
/// Conv2d chain. Asserts the emitted interval encloses the exact f64 conv range over
/// the box (plus brute-forced concrete samples), across cancellation, large-weight,
/// strided/padded, and subnormal-range configs.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_conv2d_encloses_truth_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let gpu: &dyn GpuIbpForward = &*device;
    let mut rng = sound_op_rng(0xC0_1234_5678_9ABC);
    // (in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w, wscale, box_hw)
    let configs: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        f32,
        f32,
    )] = &[
        (2, 3, 4, 4, 2, 2, 1, 1, 0, 0, 0.8, 0.3),
        (1, 2, 5, 5, 3, 3, 2, 2, 1, 1, 0.7, 0.4), // strided + padded
        (3, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1.5, 0.2), // cancellation
        (2, 2, 5, 5, 2, 2, 1, 1, 0, 0, 64.0, 0.2), // large weights (2^6)
        (1, 2, 4, 4, 2, 2, 1, 1, 0, 0, 1.0, 1e-38), // subnormal-range box
    ];
    let batch = 1usize;
    let mut checks = 0usize;
    for &(in_c, out_c, ih, iw, kh, kw, sh, sw, ph, pw, wscale, hw) in configs {
        for _ in 0..3 {
            let wlen = out_c * in_c * kh * kw;
            let weight: Vec<f32> = (0..wlen).map(|_| rng() * wscale).collect();
            let bias: Vec<f32> = (0..out_c).map(|_| rng() * 0.1).collect();
            let ilen = batch * in_c * ih * iw;
            let (mut il, mut iu) = (vec![0f32; ilen], vec![0f32; ilen]);
            for i in 0..ilen {
                let c = rng() * 0.4;
                il[i] = c - hw;
                iu[i] = c + hw;
            }
            let layer = GpuIbpLayer::Conv2d {
                weight: Arc::from(weight.clone()),
                bias: Some(Arc::from(bias.clone())),
                out_channels: out_c,
                in_channels: in_c,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                pad_h: ph,
                pad_w: pw,
                groups: 1,
                input_h: ih,
                input_w: iw,
            };
            let res = gpu
                .ibp_forward_gpu_sound(&[layer], &il, &iu, &[batch, in_c, ih, iw])
                .expect("sound GPU IBP Conv2d chain");
            let oh = (ih + 2 * ph - kh) / sh + 1;
            let ow = (iw + 2 * pw - kw) / sw + 1;
            assert_eq!(
                res.output_shape,
                vec![batch, out_c, oh, ow],
                "conv output shape"
            );
            let (lo, hi) = (&res.lower_bounds, &res.upper_bounds);
            for oc in 0..out_c {
                for y in 0..oh {
                    for x in 0..ow {
                        let mut tlo = f64::from(bias[oc]);
                        let mut thi = f64::from(bias[oc]);
                        for ic in 0..in_c {
                            for a in 0..kh {
                                for b in 0..kw {
                                    let ihp = (y * sh + a) as isize - ph as isize;
                                    let iwp = (x * sw + b) as isize - pw as isize;
                                    if ihp < 0
                                        || ihp as usize >= ih
                                        || iwp < 0
                                        || iwp as usize >= iw
                                    {
                                        continue;
                                    }
                                    let in_idx = (ic * ih + ihp as usize) * iw + iwp as usize;
                                    let w = f64::from(weight[((oc * in_c + ic) * kh + a) * kw + b]);
                                    let (xl, xu) = (f64::from(il[in_idx]), f64::from(iu[in_idx]));
                                    if w >= 0.0 {
                                        tlo += w * xl;
                                        thi += w * xu;
                                    } else {
                                        tlo += w * xu;
                                        thi += w * xl;
                                    }
                                }
                            }
                        }
                        let out_idx = (oc * oh + y) * ow + x;
                        let (glo, ghi) = (f64::from(lo[out_idx]), f64::from(hi[out_idx]));
                        assert!(
                            glo.is_finite() && ghi.is_finite() && glo <= ghi,
                            "conv interval invalid: [{glo}, {ghi}]"
                        );
                        assert!(
                            glo <= tlo,
                            "S1 conv lo violation oc{oc} y{y} x{x}: GPU_lo {glo} > truth {tlo}"
                        );
                        assert!(
                            ghi >= thi,
                            "S1 conv hi violation oc{oc} y{y} x{x}: GPU_hi {ghi} < truth {thi}"
                        );
                        checks += 1;
                    }
                }
            }
        }
    }
    assert!(checks > 0);
    eprintln!("sound GPU IBP conv2d oracle: {checks} enclosure checks, 0 violations");
}

/// S2 oracle for the SOUND Conv2d chain path (§3.2): the verdict-wired
/// `ibp_forward_gpu_sound` single-Conv2d chain is a SUPERSET of the exact CPU path it
/// replaces — `conv2d.propagate_ibp_sound` — AND the CPU reference encloses concrete
/// samples (mild config). This is the conv analogue of the Linear keystone's S2 gate,
/// validating the strict `3γ·S + 4·N·U·|endpoint|` radius against a real CPU bound.
#[test]
#[cfg(feature = "gpu-tests")]
fn sound_gpu_ibp_conv2d_superset_of_cpu_sound_t1_1() {
    let _g = gpu_test_serial_guard();
    let device = require_device();
    let gpu: &dyn GpuIbpForward = &*device;
    let mut rng = sound_op_rng(0xC0_5EED_0BAD_F00D);

    let (in_c, out_c, ih, iw, kh, kw) = (2usize, 3usize, 4usize, 4usize, 2usize, 2usize);
    let (sh, sw, ph, pw) = (1usize, 1usize, 0usize, 0usize);
    let batch = 2usize;
    for _ in 0..5 {
        let wlen = out_c * in_c * kh * kw;
        let weight: Vec<f32> = (0..wlen).map(|_| rng() * 0.6).collect();
        let bias: Vec<f32> = (0..out_c).map(|_| rng() * 0.1).collect();
        let ilen = batch * in_c * ih * iw;
        let (mut il, mut iu) = (vec![0f32; ilen], vec![0f32; ilen]);
        for i in 0..ilen {
            let c = rng() * 0.4;
            let hw = 0.15;
            il[i] = c - hw;
            iu[i] = c + hw;
        }

        // CPU reference: a single-Conv2d Network's sound IBP.
        let kernel =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, in_c, kh, kw]), weight.clone())
                .unwrap();
        let mut network = Network::new();
        network.add_layer(Layer::Conv2d(
            Conv2dLayer::with_input_shape(
                kernel,
                Some(ndarray::Array1::from(bias.clone())),
                (sh, sw),
                (ph, pw),
                ih,
                iw,
            )
            .unwrap(),
        ));
        let input = ny_tensor::BoundedTensor::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch, in_c, ih, iw]), il.clone())
                .unwrap(),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch, in_c, ih, iw]), iu.clone())
                .unwrap(),
        )
        .unwrap();
        let cpu = network
            .propagate_ibp_sound(&input)
            .expect("CPU conv sound IBP");
        let cpu_lo: Vec<f32> = cpu.lower().iter().copied().collect();
        let cpu_hi: Vec<f32> = cpu.upper().iter().copied().collect();

        // GPU sound conv via the verdict-wired chain driver.
        let layer = GpuIbpLayer::Conv2d {
            weight: Arc::from(weight.clone()),
            bias: Some(Arc::from(bias.clone())),
            out_channels: out_c,
            in_channels: in_c,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
            groups: 1,
            input_h: ih,
            input_w: iw,
        };
        let res = gpu
            .ibp_forward_gpu_sound(&[layer], &il, &iu, &[batch, in_c, ih, iw])
            .expect("sound GPU IBP Conv2d chain");
        assert_eq!(
            res.lower_bounds.len(),
            cpu_lo.len(),
            "conv S2 element parity"
        );
        assert_eq!(
            res.output_shape,
            cpu.shape().to_vec(),
            "conv S2 shape parity"
        );
        for j in 0..cpu_lo.len() {
            let (glo, ghi) = (res.lower_bounds[j], res.upper_bounds[j]);
            assert!(
                glo.is_finite() && ghi.is_finite() && glo <= ghi,
                "conv GPU interval invalid at {j}: [{glo}, {ghi}]"
            );
            assert!(
                glo <= cpu_lo[j],
                "S2 conv lo violation elem {j}: GPU_lo {glo} > CPU_lo {}",
                cpu_lo[j]
            );
            assert!(
                ghi >= cpu_hi[j],
                "S2 conv hi violation elem {j}: GPU_hi {ghi} < CPU_hi {}",
                cpu_hi[j]
            );
        }
    }
    eprintln!("sound GPU IBP conv2d S2 oracle: GPU ⊇ CPU propagate_ibp_sound, 0 violations");
}
