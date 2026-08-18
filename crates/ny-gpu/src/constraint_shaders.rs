// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! WGSL shader helpers for constraint buffer access in Clip-and-Verify.
//!
//! This module provides WGSL code fragments for accessing packed constraint data
//! from GPU buffers created by `GpuConstraintBuffers::from_cpu_buffer`.
//!
//! # Usage
//!
//! Include `CONSTRAINT_HEADER_WGSL` in your shader source to get:
//! - `ConstraintHeader` struct matching the Rust layout
//! - Helper functions for extracting packed fields
//! - Domain constraint range lookup
//!
//! ```rust,no_run
//! use ny_gpu::constraint_shaders::CONSTRAINT_HEADER_WGSL;
//!
//! let kernel_code = r#"
//! @group(0) @binding(0) var<storage, read> headers: array<ConstraintHeader>;
//! @compute @workgroup_size(64)
//! fn main() { let h = headers[0]; let len = header_data_len(h); }
//! "#;
//! let shader_source = format!("{}\n{}", CONSTRAINT_HEADER_WGSL, kernel_code);
//! ```
//!
//! # Buffer Bindings
//!
//! Shaders using these helpers should bind buffers as:
//! - `@binding(0)`: headers array
//! - `@binding(1)`: coeffs array
//! - `@binding(2)`: indices array
//! - `@binding(3)`: domain_offsets array
//!
//! # Sources
//!
//! - Design doc: `designs/2026-01-29-gpu-constraint-buffer-layout.md`
//! - Issue: #226, #256

/// WGSL struct definition and helper functions for constraint buffer access.
///
/// This string constant can be prepended to shader source code that needs
/// to access constraint buffers created by `GpuConstraintBuffers`.
///
/// # Memory Layout
///
/// The `ConstraintHeader` struct matches the Rust `ConstraintHeader` layout:
/// ```text
/// offset 0-3:   data_start (u32)
/// offset 4-5:   data_len (u16)
/// offset 6:     origin (u8)
/// offset 7:     sense (u8)
/// offset 8-11:  bias (f32)
/// offset 12-15: _padding (u32)
/// ```
///
/// In WGSL, bytes 4-7 are packed into `data_len_origin_sense` since WGSL
/// doesn't support u16 or u8 types directly. Use the helper functions
/// to extract individual fields.
pub const CONSTRAINT_HEADER_WGSL: &str = r#"
// Constraint header structure (matches Rust ny_propagate::ConstraintHeader)
// Total size: 16 bytes, aligned for efficient GPU access.
//
// Layout:
//   data_start: offset into coeffs/indices arrays
//   data_len_origin_sense: packed field
//     bits 0-15:  data_len (number of terms)
//     bits 16-23: origin (0=Split, 1=Output, 2=BoundProp)
//     bits 24-31: sense (0=Le, 1=Ge)
//   bias: right-hand side constant
//   _padding: alignment padding
struct ConstraintHeader {
    data_start: u32,
    data_len_origin_sense: u32,
    bias: f32,
    _padding: u32,
}

// Constraint sense enum values
const SENSE_LE: u32 = 0u;  // Less than or equal (≤)
const SENSE_GE: u32 = 1u;  // Greater than or equal (≥)

// Constraint origin enum values
const ORIGIN_SPLIT: u32 = 0u;     // From ReLU/activation split
const ORIGIN_OUTPUT: u32 = 1u;    // From output property
const ORIGIN_BOUNDPROP: u32 = 2u; // From bound propagation

// Extract data_len (number of terms) from packed field
fn header_data_len(h: ConstraintHeader) -> u32 {
    return h.data_len_origin_sense & 0xFFFFu;
}

// Extract origin from packed field
fn header_origin(h: ConstraintHeader) -> u32 {
    return (h.data_len_origin_sense >> 16u) & 0xFFu;
}

// Extract sense from packed field
fn header_sense(h: ConstraintHeader) -> u32 {
    return (h.data_len_origin_sense >> 24u) & 0xFFu;
}

// ============================================================================
// USAGE PATTERNS FOR CONSTRAINT BUFFER ACCESS
// ============================================================================
//
// WGSL does not allow passing storage buffer pointers as function arguments.
// Instead, access the global bindings directly. Here are the recommended patterns:
//
// 1. Declare global bindings in your shader:
//    @group(0) @binding(0) var<storage, read> constraint_headers: array<ConstraintHeader>;
//    @group(0) @binding(1) var<storage, read> constraint_coeffs: array<f32>;
//    @group(0) @binding(2) var<storage, read> constraint_indices: array<u32>;
//    @group(0) @binding(3) var<storage, read> constraint_domain_offsets: array<u32>;
//
// 2. Get domain constraint range (inline):
//    let range_start = constraint_domain_offsets[domain_idx];
//    let range_end = constraint_domain_offsets[domain_idx + 1u];
//    // Then iterate: for (var i = range_start; i < range_end; i++) { ... }
//
// 3. Evaluate constraint (inline):
//    let h = constraint_headers[constraint_idx];
//    let data_len = header_data_len(h);
//    var sum: f32 = 0.0;
//    for (var i: u32 = 0u; i < data_len; i = i + 1u) {
//        let idx = constraint_indices[h.data_start + i];
//        let coeff = constraint_coeffs[h.data_start + i];
//        sum = sum + coeff * x[idx];  // x is your input array
//    }
//    // For Le constraints: violation = sum - h.bias > 0
//    // For Ge constraints: violation = h.bias - sum > 0
//    let violation = select(sum - h.bias, h.bias - sum, header_sense(h) == SENSE_GE);
// ============================================================================
"#;

/// Bind group layout entries for constraint buffers.
///
/// Use these entries when creating a bind group layout that includes
/// constraint buffer access. These correspond to the bindings expected
/// by `CONSTRAINT_HEADER_WGSL`.
pub const CONSTRAINT_BUFFER_LAYOUT_ENTRIES: [wgpu::BindGroupLayoutEntry; 4] = [
    // @binding(0): headers array
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // @binding(1): coeffs array
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // @binding(2): indices array
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // @binding(3): domain_offsets array
    wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgsl_contains_expected_definitions() {
        // Verify the WGSL fragment contains expected struct and function definitions
        assert!(CONSTRAINT_HEADER_WGSL.contains("struct ConstraintHeader"));
        assert!(CONSTRAINT_HEADER_WGSL.contains("fn header_data_len"));
        assert!(CONSTRAINT_HEADER_WGSL.contains("fn header_origin"));
        assert!(CONSTRAINT_HEADER_WGSL.contains("fn header_sense"));
        // Usage patterns are now documented as comments, not functions
        assert!(CONSTRAINT_HEADER_WGSL.contains("USAGE PATTERNS"));
        assert!(CONSTRAINT_HEADER_WGSL.contains("constraint_domain_offsets"));
    }

    #[test]
    fn test_wgsl_compiles_with_bindings() {
        // Verify the WGSL fragment compiles when combined with required bindings
        // This catches invalid WGSL syntax that string checks would miss
        let full_shader = format!(
            r#"
{}

@group(0) @binding(0) var<storage, read> constraint_headers: array<ConstraintHeader>;
@group(0) @binding(1) var<storage, read> constraint_coeffs: array<f32>;
@group(0) @binding(2) var<storage, read> constraint_indices: array<u32>;
@group(0) @binding(3) var<storage, read> constraint_domain_offsets: array<u32>;
@group(1) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let domain_idx = gid.x;

    // Test domain range lookup (inline pattern)
    let range_start = constraint_domain_offsets[domain_idx];
    let range_end = constraint_domain_offsets[domain_idx + 1u];

    // Test header access and field extraction
    if (range_start < range_end) {{
        let h = constraint_headers[range_start];
        let data_len = header_data_len(h);
        let origin = header_origin(h);
        let sense = header_sense(h);

        // Test coefficient/index access (inline pattern)
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < data_len; i = i + 1u) {{
            let coeff = constraint_coeffs[h.data_start + i];
            sum = sum + coeff;
        }}

        // Use origin and sense to avoid dead code (verify extraction works)
        // Origin: 0=Split, 1=Output, 2=BoundProp; Sense: 0=Le, 1=Ge
        let sign = select(1.0, -1.0, sense == SENSE_GE);
        output[domain_idx] = sum * sign + f32(origin);
    }}
}}
"#,
            CONSTRAINT_HEADER_WGSL
        );

        naga::front::wgsl::parse_str(&full_shader)
            .unwrap_or_else(|error| panic!("constraint shader must parse as WGSL: {error}"));
    }

    #[test]
    fn test_layout_entries_count() {
        assert_eq!(CONSTRAINT_BUFFER_LAYOUT_ENTRIES.len(), 4);
        // Verify binding indices are 0, 1, 2, 3
        for (i, entry) in CONSTRAINT_BUFFER_LAYOUT_ENTRIES.iter().enumerate() {
            assert_eq!(entry.binding, i as u32);
        }
    }
}
