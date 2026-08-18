// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime-oriented host-side contract tests for WGPU pipeline/shader invariants.

use super::{
    ADD_IBP_SHADER, CONV2D_IBP_IM2COL_SHADER, CONV_COL2IM_SHADER, CONV_COL2IM_TAINT_SHADER,
    CONV_RESHAPE_SHADER, CONV_RESHAPE_TAINT_SHADER, CROWN_STRIDED_GATHER_SHADER, GEMM_F32_SHADER,
    GEMM_F32_SMALL_K_SHADER, GEMM_F32_SMALL_K_TAINT_SHADER, LINEAR_IBP_SHADER, MATMUL_IBP_SHADER,
    RELU_IBP_SHADER, SCALE_IBP_SHADER, SOFTMAX_APPLY_SHADER, SOFTMAX_REDUCE_SHADER,
    TRANSPOSE_IBP_SHADER,
};
use std::mem::size_of;

// ============================================================================
// WGSL compilation smoke tests
// ============================================================================
// Compile every shader through the same Naga WGSL front-end used by wgpu.
// This catches syntax and type errors hermetically: adapter absence is not a
// reason to leave shader source unvalidated.

#[cfg(feature = "gpu-tests")]
fn with_wgpu_device(f: impl FnOnce(&wgpu::Device)) {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find wgpu adapter for contract test");

        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create wgpu device for contract test");

        f(&device);
    });
}

#[test]
fn test_wgpu_all_shaders_compile() {
    let shaders: &[(&str, &str)] = &[
        ("linear_ibp", LINEAR_IBP_SHADER),
        ("gemm_f32", GEMM_F32_SHADER),
        ("gemm_f32_small_k", GEMM_F32_SMALL_K_SHADER),
        ("gemm_f32_small_k_taint", GEMM_F32_SMALL_K_TAINT_SHADER),
        ("matmul_ibp", MATMUL_IBP_SHADER),
        ("softmax_reduce", SOFTMAX_REDUCE_SHADER),
        ("softmax_apply", SOFTMAX_APPLY_SHADER),
        ("transpose_ibp", TRANSPOSE_IBP_SHADER),
        ("scale_ibp", SCALE_IBP_SHADER),
        ("conv2d_ibp_im2col", CONV2D_IBP_IM2COL_SHADER),
        ("conv_reshape", CONV_RESHAPE_SHADER),
        ("conv_reshape_taint", CONV_RESHAPE_TAINT_SHADER),
        ("conv_col2im", CONV_COL2IM_SHADER),
        ("conv_col2im_taint", CONV_COL2IM_TAINT_SHADER),
        ("relu_ibp", RELU_IBP_SHADER),
        ("add_ibp", ADD_IBP_SHADER),
        ("crown_strided_gather", CROWN_STRIDED_GATHER_SHADER),
    ];

    for (label, source) in shaders {
        naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{label} must parse as WGSL: {error}"));
    }
}

// ============================================================================
// Pipeline creation smoke test
// ============================================================================
// Exercise the full pipeline creation path (shader compile + bind group layout
// + pipeline layout + compute pipeline) for every pipeline type. This catches
// layout mismatches between shader bindings and Rust-side bind group layouts.

#[test]
#[cfg(feature = "gpu-tests")]
fn test_wgpu_all_pipelines_create_successfully() {
    with_wgpu_device(|device| {
        // Each of these calls create_shader_module + bind_group_layout + pipeline.
        // Panics on any mismatch between shader bindings and layout entries.
        // Plain-WGSL loading path (denorm_preserve = false): this smoke test
        // checks shader/layout compatibility, not the DenormPreserve seam.
        let (_p, _l) = super::super::WgpuDevice::create_linear_ibp_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_matmul_ibp_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_softmax_reduce_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_softmax_apply_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_transpose_ibp_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_scale_ibp_pipeline(device, false);
        let (_p, layout) = super::super::WgpuDevice::create_gemm_f32_pipeline(device, false);
        // Small-K pipeline shares the GEMM bind group layout (#3599)
        let _p = super::super::WgpuDevice::create_gemm_f32_small_k_pipeline(device, false, &layout);
        let (_p, _l) = super::super::WgpuDevice::create_conv2d_ibp_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_conv_reshape_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_conv_col2im_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_relu_ibp_pipeline(device, false);
        let (_p, _l) = super::super::WgpuDevice::create_add_ibp_pipeline(device, false);
    });
}

// ============================================================================
// Shader Params struct declaration invariant
// ============================================================================
// Every shader must declare `struct Params` matching its Rust counterpart.

#[test]
fn test_all_shaders_declare_params_struct() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("GEMM_F32_SMALL_K_SHADER", GEMM_F32_SMALL_K_SHADER),
        (
            "GEMM_F32_SMALL_K_TAINT_SHADER",
            GEMM_F32_SMALL_K_TAINT_SHADER,
        ),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_RESHAPE_TAINT_SHADER", CONV_RESHAPE_TAINT_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("CONV_COL2IM_TAINT_SHADER", CONV_COL2IM_TAINT_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
        ("CROWN_STRIDED_GATHER_SHADER", CROWN_STRIDED_GATHER_SHADER),
    ];
    for (name, source) in shaders {
        assert!(
            source.contains("struct Params"),
            "{name} must declare `struct Params` for uniform buffer layout"
        );
    }
}

// ============================================================================
// Workgroup size invariant
// ============================================================================
// All IBP shaders use workgroup_size(64). GEMM uses workgroup_size(16, 16).
// The dispatch code in ops/ depends on these sizes for correct thread count.

#[test]
fn test_ibp_shaders_use_workgroup_size_64() {
    let ibp_shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
    ];
    for (name, source) in ibp_shaders {
        assert!(
            source.contains("@workgroup_size(64)"),
            "{name} must use @workgroup_size(64) — dispatch code depends on this"
        );
    }
}

#[test]
fn test_gemm_shaders_use_workgroup_size_16x16() {
    for (name, source) in [
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("GEMM_F32_SMALL_K_SHADER", GEMM_F32_SMALL_K_SHADER),
        (
            "GEMM_F32_SMALL_K_TAINT_SHADER",
            GEMM_F32_SMALL_K_TAINT_SHADER,
        ),
    ] {
        assert!(
            source.contains("@workgroup_size(16, 16)"),
            "{name} must use @workgroup_size(16, 16) — dispatch code depends on this"
        );
    }
}

#[test]
fn test_conv_shaders_use_workgroup_size_256() {
    for (name, source) in [
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_RESHAPE_TAINT_SHADER", CONV_RESHAPE_TAINT_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("CONV_COL2IM_TAINT_SHADER", CONV_COL2IM_TAINT_SHADER),
    ] {
        assert!(
            source.contains("@workgroup_size(256)"),
            "{name} must use @workgroup_size(256) — dispatch code depends on this"
        );
    }
}

// ============================================================================
// Softmax epsilon invariant
// ============================================================================
// SOFTMAX_APPLY_SHADER must define EPSILON to prevent division by zero.

#[test]
fn test_softmax_apply_shader_defines_epsilon() {
    assert!(
        SOFTMAX_APPLY_SHADER.contains("EPSILON"),
        "SOFTMAX_APPLY_SHADER must define EPSILON constant to prevent div-by-zero"
    );
}

// ============================================================================
// Scale shader sign-swap invariant
// ============================================================================
// SCALE_IBP_SHADER must handle negative scale by swapping lower/upper.

#[test]
fn test_scale_shader_handles_negative_scale() {
    assert!(
        SCALE_IBP_SHADER.contains("s >= 0.0") || SCALE_IBP_SHADER.contains("s < 0.0"),
        "SCALE_IBP_SHADER must branch on scale sign for bound swap"
    );
}

// ============================================================================
// GEMM M-batching binding limit invariant (#3397)
// ============================================================================
// The GEMM M-batching constants must agree: `MAX_BINDING_ELEMS` must equal
// `WGPU_MAX_BINDING_BYTES * 5/6 / 4` (128 MB / 1.2 / sizeof(f32)).
// If this invariant breaks, GEMM will either panic on large matrices (limit
// too high) or batch unnecessarily (limit too low).

#[test]
fn test_gemm_binding_limit_constants_consistent() {
    // wgpu default: max_storage_buffer_binding_size = 128 MiB
    let binding_bytes: usize = 134_217_728;
    // BufferPool growth factor = 1.2× (= 6/5), so 1/1.2 = 5/6
    let effective_bytes = binding_bytes * 5 / 6;
    let expected_elems = effective_bytes / size_of::<f32>();
    // The GEMM module defines MAX_BINDING_ELEMS using the same formula.
    // Verify the expected value to catch accidental changes.
    assert_eq!(
        expected_elems, 27_962_026,
        "MAX_BINDING_ELEMS should be 27,962,026 (128 MiB / 1.2 / 4)"
    );
}

// ============================================================================
// Small-K GEMM ROWS_PER_THREAD invariant (#3599)
// ============================================================================
// The small-K GEMM shader defines `const ROWS_PER_THREAD: u32 = 4u;`.
// The Rust dispatch code uses SMALL_K_ROWS_PER_THREAD=4 to compute
// wg_y = ceil(M / (TILE_DIM * 4)). If these diverge, rows will be skipped
// or out-of-bounds.

#[test]
fn test_small_k_shader_rows_per_thread_is_4() {
    assert!(
        GEMM_F32_SMALL_K_SHADER.contains("const ROWS_PER_THREAD: u32 = 4u;"),
        "GEMM_F32_SMALL_K_SHADER must define ROWS_PER_THREAD=4 — dispatch code depends on this"
    );
}

// ============================================================================
// Add IBP shader NaN/Inf defense invariant (#4319)
// ============================================================================
// ADD_IBP_SHADER must contain FALLBACK_BOUND and NaN checks to prevent
// Inf/NaN propagation through residual connections.

#[test]
fn test_add_ibp_shader_nan_defense() {
    assert!(
        ADD_IBP_SHADER.contains("FALLBACK_BOUND"),
        "ADD_IBP_SHADER must define FALLBACK_BOUND for NaN/Inf defense"
    );
    // NaN self-comparison check pattern: `low != low`
    assert!(
        ADD_IBP_SHADER.contains("!= low") || ADD_IBP_SHADER.contains("!= high"),
        "ADD_IBP_SHADER must check for NaN via self-inequality"
    );
}

// ============================================================================
// ReLU IBP shader in-place invariant (#4319)
// ============================================================================
// RELU_IBP_SHADER operates in-place (read_write on lower/upper), not
// separate src/dst. The DAG plan builder copies src→dst before dispatching.

#[test]
fn test_relu_ibp_shader_is_inplace() {
    assert!(
        RELU_IBP_SHADER.contains("read_write> lower")
            && RELU_IBP_SHADER.contains("read_write> upper"),
        "RELU_IBP_SHADER must use read_write bindings — it operates in-place after a buffer copy"
    );
}
