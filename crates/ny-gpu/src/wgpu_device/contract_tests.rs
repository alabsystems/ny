// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host-side contract tests for WGPU pipeline/shader invariants.
//!
//! These tests run in the default test suite (no `gpu-tests` feature required).
//! They verify structural invariants that prevent silent runtime failures:
//! - Param struct sizes match WGSL uniform expectations (16 bytes each)
//! - Shader binding declarations match pipeline bind group layouts
//! - All shaders define a `fn main` entry point
//! - WGSL source compiles on the host adapter
//! - NaN safety constants are consistent across shaders
//!
//! Part of #1908.

use super::params::{
    AddIbpParams, Conv2dIbpParams, ConvCol2imParams, ConvReshapeParams, GemmParams,
    LinearIbpParams, MatmulIbpParams, ReluIbpParams, ScaleIbpParams, SoftmaxIbpParams,
    TransposeIbpParams,
};
use super::shaders::{
    ADD_IBP_SHADER, CONV2D_IBP_IM2COL_SHADER, CONV_COL2IM_SHADER, CONV_RESHAPE_SHADER,
    CROWN_CONCRETIZE_SHADER, GEMM_F32_SHADER, GEMM_F32_SMALL_K_SHADER, LINEAR_IBP_SHADER,
    MATMUL_IBP_SHADER, RELU_IBP_SHADER, SCALE_IBP_SHADER, SOFTMAX_APPLY_SHADER,
    SOFTMAX_REDUCE_SHADER, TRANSPOSE_IBP_SHADER,
};
use std::mem::size_of;

// ============================================================================
// Param struct size invariants
// ============================================================================
// Every params struct is passed as a uniform buffer to WGSL shaders.
// WGSL struct Params has 4 u32 fields = 16 bytes. If the Rust struct
// drifts from this, the GPU will read garbage.

#[test]
fn test_linear_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<LinearIbpParams>(),
        16,
        "LinearIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_matmul_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<MatmulIbpParams>(),
        16,
        "MatmulIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_gemm_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<GemmParams>(),
        16,
        "GemmParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_softmax_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<SoftmaxIbpParams>(),
        16,
        "SoftmaxIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_transpose_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<TransposeIbpParams>(),
        16,
        "TransposeIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_scale_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<ScaleIbpParams>(),
        16,
        "ScaleIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_conv_reshape_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<ConvReshapeParams>(),
        16,
        "ConvReshapeParams must be exactly 16 bytes to match WGSL Params struct (4 x u32)"
    );
}

#[test]
fn test_conv2d_ibp_params_size_is_64_bytes() {
    assert_eq!(
        size_of::<Conv2dIbpParams>(),
        64,
        "Conv2dIbpParams must be exactly 64 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_conv_col2im_params_size_is_64_bytes() {
    assert_eq!(
        size_of::<ConvCol2imParams>(),
        64,
        "ConvCol2imParams must be exactly 64 bytes to match WGSL Params struct (14 x u32 + vec2)"
    );
}

#[test]
fn test_relu_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<ReluIbpParams>(),
        16,
        "ReluIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

#[test]
fn test_add_ibp_params_size_is_16_bytes() {
    assert_eq!(
        size_of::<AddIbpParams>(),
        16,
        "AddIbpParams must be exactly 16 bytes to match WGSL Params struct"
    );
}

// ============================================================================
// Shader entry point invariant
// ============================================================================
// Every compute pipeline specifies `entry_point: Some("main")`.
// If a shader doesn't define `fn main`, pipeline creation will fail at runtime.

#[test]
fn test_all_shaders_define_main_entry_point() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
    ];
    for (name, source) in shaders {
        assert!(
            source.contains("fn main("),
            "{name} must define `fn main(` entry point"
        );
    }
}

// ============================================================================
// Shader binding count invariants
// ============================================================================
// Each pipeline's bind group layout must match the number of @binding(N)
// declarations in the WGSL shader. Drift between these causes runtime panics.

/// Count the number of `@binding(` declarations in a WGSL shader source.
fn count_bindings(shader_source: &str) -> usize {
    shader_source.matches("@binding(").count()
}

/// Extract the maximum binding index from a WGSL shader source.
fn max_binding_index(shader_source: &str) -> Option<u32> {
    let mut max_idx = None;
    for (i, _) in shader_source.match_indices("@binding(") {
        let after = &shader_source[i + "@binding(".len()..];
        if let Some(end) = after.find(')') {
            if let Ok(idx) = after[..end].trim().parse::<u32>() {
                max_idx = Some(max_idx.map_or(idx, |m: u32| m.max(idx)));
            }
        }
    }
    max_idx
}

#[test]
fn test_linear_ibp_shader_has_8_bindings() {
    // pipelines.rs: create_linear_ibp_pipeline has 8 entries (0..7)
    assert_eq!(count_bindings(LINEAR_IBP_SHADER), 8);
    assert_eq!(max_binding_index(LINEAR_IBP_SHADER), Some(7));
}

#[test]
fn test_matmul_ibp_shader_has_7_bindings() {
    // pipelines.rs: create_matmul_ibp_pipeline has 7 entries (0..6)
    assert_eq!(count_bindings(MATMUL_IBP_SHADER), 7);
    assert_eq!(max_binding_index(MATMUL_IBP_SHADER), Some(6));
}

#[test]
fn test_softmax_reduce_shader_has_8_bindings() {
    // pipelines.rs: create_softmax_reduce_pipeline has 8 entries (0..7)
    assert_eq!(count_bindings(SOFTMAX_REDUCE_SHADER), 8);
    assert_eq!(max_binding_index(SOFTMAX_REDUCE_SHADER), Some(7));
}

#[test]
fn test_softmax_apply_shader_has_7_bindings() {
    // pipelines.rs: create_softmax_apply_pipeline has 7 entries (0..6)
    assert_eq!(count_bindings(SOFTMAX_APPLY_SHADER), 7);
    assert_eq!(max_binding_index(SOFTMAX_APPLY_SHADER), Some(6));
}

#[test]
fn test_transpose_ibp_shader_has_5_bindings() {
    // pipelines.rs: create_transpose_ibp_pipeline has 5 entries (0..4)
    assert_eq!(count_bindings(TRANSPOSE_IBP_SHADER), 5);
    assert_eq!(max_binding_index(TRANSPOSE_IBP_SHADER), Some(4));
}

#[test]
fn test_scale_ibp_shader_has_5_bindings() {
    // pipelines.rs: create_scale_ibp_pipeline has 5 entries (0..4)
    assert_eq!(count_bindings(SCALE_IBP_SHADER), 5);
    assert_eq!(max_binding_index(SCALE_IBP_SHADER), Some(4));
}

#[test]
fn test_gemm_f32_shader_has_4_bindings() {
    // pipelines.rs: create_gemm_f32_pipeline has 4 entries (0..3)
    assert_eq!(count_bindings(GEMM_F32_SHADER), 4);
    assert_eq!(max_binding_index(GEMM_F32_SHADER), Some(3));
}

#[test]
fn test_conv2d_ibp_shader_has_8_bindings() {
    assert_eq!(count_bindings(CONV2D_IBP_IM2COL_SHADER), 8);
    assert_eq!(max_binding_index(CONV2D_IBP_IM2COL_SHADER), Some(7));
}

#[test]
fn test_conv_reshape_shader_has_3_bindings() {
    // conv_pipelines.rs: create_conv_reshape_pipeline has 3 entries (0..2)
    assert_eq!(count_bindings(CONV_RESHAPE_SHADER), 3);
    assert_eq!(max_binding_index(CONV_RESHAPE_SHADER), Some(2));
}

#[test]
fn test_conv_col2im_shader_has_3_bindings() {
    // conv_pipelines.rs: create_conv_col2im_pipeline has 3 entries (0..2)
    assert_eq!(count_bindings(CONV_COL2IM_SHADER), 3);
    assert_eq!(max_binding_index(CONV_COL2IM_SHADER), Some(2));
}

#[test]
fn test_relu_ibp_shader_has_3_bindings() {
    // pipelines.rs: create_relu_ibp_pipeline has 3 entries (0..2)
    assert_eq!(count_bindings(RELU_IBP_SHADER), 3);
    assert_eq!(max_binding_index(RELU_IBP_SHADER), Some(2));
}

#[test]
fn test_add_ibp_shader_has_7_bindings() {
    // pipelines.rs: create_add_ibp_pipeline has 7 entries (0..6)
    assert_eq!(count_bindings(ADD_IBP_SHADER), 7);
    assert_eq!(max_binding_index(ADD_IBP_SHADER), Some(6));
}

// ============================================================================
// Binding index sequential invariant
// ============================================================================
// All pipelines use sequential binding indices starting from 0 with no gaps.
// Missing indices cause wgpu validation errors at pipeline creation.

/// Verify that binding indices in a shader are sequential 0..N.
fn verify_sequential_bindings(shader_source: &str, shader_name: &str) {
    let mut indices = Vec::new();
    for (i, _) in shader_source.match_indices("@binding(") {
        let after = &shader_source[i + "@binding(".len()..];
        if let Some(end) = after.find(')') {
            if let Ok(idx) = after[..end].trim().parse::<u32>() {
                indices.push(idx);
            }
        }
    }
    indices.sort_unstable();
    indices.dedup();
    let expected: Vec<u32> = (0..indices.len() as u32).collect();
    assert_eq!(
        indices,
        expected,
        "{shader_name}: binding indices must be sequential 0..{}, got {indices:?}",
        indices.len()
    );
}

#[test]
fn test_all_shaders_have_sequential_binding_indices() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
    ];
    for (name, source) in shaders {
        verify_sequential_bindings(source, name);
    }
}

// ============================================================================
// Binding 0 is uniform invariant
// ============================================================================
// Every pipeline has binding 0 as a uniform buffer for the Params struct.

#[test]
fn test_all_shaders_use_binding_0_as_uniform() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SOFTMAX_APPLY_SHADER", SOFTMAX_APPLY_SHADER),
        ("TRANSPOSE_IBP_SHADER", TRANSPOSE_IBP_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("CONV_RESHAPE_SHADER", CONV_RESHAPE_SHADER),
        ("CONV_COL2IM_SHADER", CONV_COL2IM_SHADER),
        ("RELU_IBP_SHADER", RELU_IBP_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
    ];
    for (name, source) in shaders {
        assert!(
            source.contains("@binding(0) var<uniform>"),
            "{name}: binding 0 must be var<uniform> for Params struct"
        );
    }
}

// ============================================================================
// Fallback bound constant consistency (#2258, #2390)
// ============================================================================
// All shaders that perform arithmetic on bounds define FALLBACK_BOUND.
// These must match crate::FALLBACK_BOUND to ensure GPU and CPU paths produce
// consistent sanitization behavior.

#[test]
fn test_fallback_bound_consistent_across_shaders() {
    let shaders: &[(&str, &str)] = &[
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("GEMM_F32_SHADER", GEMM_F32_SHADER),
        ("SOFTMAX_REDUCE_SHADER", SOFTMAX_REDUCE_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("ADD_IBP_SHADER", ADD_IBP_SHADER),
    ];
    // WGSL literal must match the Rust constant (1e10).
    let expected = "1e10";
    for (name, source) in shaders {
        assert!(
            source.contains(expected),
            "{name} must define FALLBACK_BOUND = {expected} (must match crate::FALLBACK_BOUND)"
        );
    }
}

#[test]
fn test_nan_safety_functions_present_in_ibp_shaders() {
    // IBP shaders that compute bounds via multiplication/accumulation must have
    // is_non_finite + nan_safe_lower + nan_safe_upper guards.
    // Part of #2390: added SCALE_IBP_SHADER.
    for (name, source) in [
        ("LINEAR_IBP_SHADER", LINEAR_IBP_SHADER),
        ("MATMUL_IBP_SHADER", MATMUL_IBP_SHADER),
        ("CONV2D_IBP_IM2COL_SHADER", CONV2D_IBP_IM2COL_SHADER),
        ("SCALE_IBP_SHADER", SCALE_IBP_SHADER),
    ] {
        assert!(
            source.contains("fn is_non_finite("),
            "{name} must define is_non_finite helper"
        );
        assert!(
            source.contains("fn nan_safe_lower("),
            "{name} must define nan_safe_lower helper"
        );
        assert!(
            source.contains("fn nan_safe_upper("),
            "{name} must define nan_safe_upper helper"
        );
    }
}

#[test]
fn test_nan_safety_functions_present_in_softmax_shaders() {
    // SOFTMAX_REDUCE_SHADER: must guard max reduction (max(x, NaN) is
    // implementation-defined in WGSL) and exp output overflow.
    // Part of #2390.
    assert!(
        SOFTMAX_REDUCE_SHADER.contains("fn is_non_finite("),
        "SOFTMAX_REDUCE_SHADER must define is_non_finite helper"
    );
    assert!(
        SOFTMAX_REDUCE_SHADER.contains("fn safe_exp("),
        "SOFTMAX_REDUCE_SHADER must define safe_exp helper for guarded exponentiation"
    );

    // SOFTMAX_APPLY_SHADER: must guard division output to catch NaN/Inf
    // and clamp probability bounds to [0, 1].
    // Part of #2390.
    assert!(
        SOFTMAX_APPLY_SHADER.contains("fn is_non_finite("),
        "SOFTMAX_APPLY_SHADER must define is_non_finite helper"
    );
}

#[test]
fn test_nan_safety_functions_present_in_gemm_shader() {
    // GEMM shader uses nan_safe_clamp (NaN → preserved for downstream detection,
    // Inf → clamp to ±FALLBACK_BOUND) via x != x check. #2708: NaN is no longer
    // replaced with 0.0 (which was unsound — silently drops coefficient contribution).
    // Part of #2366, #2708.
    assert!(
        GEMM_F32_SHADER.contains("fn nan_safe_clamp("),
        "GEMM_F32_SHADER must define nan_safe_clamp helper"
    );
    // Verify the NaN-specific check pattern (x != x distinguishes NaN from Inf).
    assert!(
        GEMM_F32_SHADER.contains("x != x"),
        "GEMM_F32_SHADER nan_safe_clamp must use x != x for NaN detection"
    );
}

#[test]
fn test_crown_concretize_shader_degrades_fallback_bound_coefficients_2708() {
    assert!(
        CROWN_CONCRETIZE_SHADER.contains("abs(a_l) >= FALLBACK_BOUND"),
        "CROWN_CONCRETIZE_SHADER must treat exact +/ -FALLBACK_BOUND lower coefficients as degraded"
    );
    assert!(
        CROWN_CONCRETIZE_SHADER.contains("abs(a_u) >= FALLBACK_BOUND"),
        "CROWN_CONCRETIZE_SHADER must treat exact +/ -FALLBACK_BOUND upper coefficients as degraded"
    );
}

#[path = "contract_plan_cache_tests.rs"]
mod contract_plan_cache_tests;
#[path = "contract_runtime_tests.rs"]
mod contract_runtime_tests;
