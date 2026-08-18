// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Full-IEEE kernels and pipeline bundle for worded intermediate-sweep DAG
//! carriers.
//!
//! These kernels are not an extension of `RESIDENT_SEG_MERGE_SHADER`. That
//! legacy shader has no word channel and its error expression is explicitly
//! outside the verdict surface. Here every nonnegative merge term is assembled
//! with a staged outward step, coefficient/error words are OR-preserved, bias
//! births enter the per-row accumulator, and Sub negation is a sign-bit XOR.
//!
//! The merge proof currently assumes the shader-loading profile authenticated
//! by full IEEE sound-GPU authority. Charged/flush-only devices must decline
//! before these pipelines are created or any carrier is allocated.

/// Compact `(carrier_row, identity_coordinate)` upload entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::wgpu_device) struct SweepInjectResetGpu {
    pub(in crate::wgpu_device) carrier_row: u32,
    pub(in crate::wgpu_device) coordinate: u32,
}

/// Uniform for the two reset kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::wgpu_device) struct SweepInjectParams {
    pub(in crate::wgpu_device) reset_count: u32,
    pub(in crate::wgpu_device) rows: u32,
    pub(in crate::wgpu_device) dim: u32,
    pub(in crate::wgpu_device) total: u32,
    pub(in crate::wgpu_device) stride: u32,
    pub(in crate::wgpu_device) _padding: [u32; 3],
}

/// Uniform for elementwise negate and merge kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::wgpu_device) struct SweepElementParams {
    pub(in crate::wgpu_device) n: u32,
    pub(in crate::wgpu_device) stride: u32,
    /// Matrix column count for row-word births; set to one for row kernels.
    pub(in crate::wgpu_device) dim: u32,
    pub(in crate::wgpu_device) _padding: u32,
}

/// Reset coefficient/error/bias value lanes for the identity rows injected at
/// one slot. One thread owns one `(reset, coordinate)`, so different resets can
/// never race: carrier rows are globally unique across target descriptors.
pub(in crate::wgpu_device) const SWEEP_INJECT_VALUES_SHADER: &str = r#"
struct Params {
    reset_count: u32,
    rows: u32,
    dim: u32,
    total: u32,
    stride: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> resets: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> lower_center: array<f32>;
@group(0) @binding(3) var<storage, read_write> upper_center: array<f32>;
@group(0) @binding(4) var<storage, read_write> lower_radius: array<f32>;
@group(0) @binding(5) var<storage, read_write> upper_radius: array<f32>;
@group(0) @binding(6) var<storage, read_write> lower_bias: array<f32>;
@group(0) @binding(7) var<storage, read_write> upper_bias: array<f32>;
@group(0) @binding(8) var<storage, read_write> lower_bias_radius: array<f32>;
@group(0) @binding(9) var<storage, read_write> upper_bias_radius: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.stride == 0u) { return; }
    var flat = gid.x;
    loop {
        if (flat >= p.total) { break; }
        let reset_index = flat / p.dim;
        let column = flat % p.dim;
        if (reset_index < p.reset_count) {
            let reset = resets[reset_index];
            let row = reset.x;
            let coordinate = reset.y;
            if (row < p.rows && coordinate < p.dim) {
                let index = row * p.dim + column;
                let identity = select(0.0, 1.0, column == coordinate);
                lower_center[index] = identity;
                upper_center[index] = identity;
                lower_radius[index] = 0.0;
                upper_radius[index] = 0.0;
                if (column == 0u) {
                    lower_bias[row] = 0.0;
                    upper_bias[row] = 0.0;
                    lower_bias_radius[row] = 0.0;
                    upper_bias_radius[row] = 0.0;
                }
            }
        }
        flat += p.stride;
    }
}
"#;

/// Reset all four per-entry word twins and the sticky row accumulator for the
/// same injected identity rows. Split from the value reset to stay below the
/// portable storage-buffer binding limit.
pub(in crate::wgpu_device) const SWEEP_INJECT_WORDS_SHADER: &str = r#"
struct Params {
    reset_count: u32,
    rows: u32,
    dim: u32,
    total: u32,
    stride: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> resets: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> lower_center_word: array<u32>;
@group(0) @binding(3) var<storage, read_write> upper_center_word: array<u32>;
@group(0) @binding(4) var<storage, read_write> lower_radius_word: array<u32>;
@group(0) @binding(5) var<storage, read_write> upper_radius_word: array<u32>;
@group(0) @binding(6) var<storage, read_write> taint_rows: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.stride == 0u) { return; }
    var flat = gid.x;
    loop {
        if (flat >= p.total) { break; }
        let reset_index = flat / p.dim;
        let column = flat % p.dim;
        if (reset_index < p.reset_count) {
            let row = resets[reset_index].x;
            if (row < p.rows) {
                let index = row * p.dim + column;
                lower_center_word[index] = 0u;
                upper_center_word[index] = 0u;
                lower_radius_word[index] = 0u;
                upper_radius_word[index] = 0u;
                if (column == 0u) { taint_rows[row] = 0u; }
            }
        }
        flat += p.stride;
    }
}
"#;

/// Exact Sub-rhs coefficient transform. Sign-bit XOR avoids an arithmetic
/// operation entirely and intentionally does not swap lower/upper lanes.
pub(in crate::wgpu_device) const SWEEP_NEGATE_CENTERS_SHADER: &str = r#"
struct Params { n: u32, stride: u32, dim: u32, _padding: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> lower_center: array<f32>;
@group(0) @binding(2) var<storage, read_write> upper_center: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.stride == 0u) { return; }
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        lower_center[i] = bitcast<f32>(bitcast<u32>(lower_center[i]) ^ 0x80000000u);
        upper_center[i] = bitcast<f32>(bitcast<u32>(upper_center[i]) ^ 0x80000000u);
        i += p.stride;
    }
}
"#;

/// Merge one lower or upper coefficient centre/radius pair and both word twins.
///
/// For exact `z = a + b`, full IEEE RN gives `s = RN(z)`. In the normal range,
/// `|s-z| <= gamma_1 |s|`, where `gamma_1 = u/(1-u)`; in the subnormal/lowest
/// normal bin an exact sum of two f32 values remains an exact multiple of
/// `2^-149`, so the add has zero gap. The upward-rounded `rho` therefore covers
/// the centre gap. Every nonnegative error addition is upward-stepped before
/// the next term is introduced, so no previously certified radius is swallowed.
pub(in crate::wgpu_device) const SWEEP_MERGE_MATRIX_SHADER: &str = r#"
struct Params { n: u32, stride: u32, dim: u32, _padding: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> dst_center: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_radius: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_center_word: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_radius_word: array<u32>;
@group(0) @binding(5) var<storage, read> src_center: array<f32>;
@group(0) @binding(6) var<storage, read> src_radius: array<f32>;
@group(0) @binding(7) var<storage, read> src_center_word: array<u32>;
@group(0) @binding(8) var<storage, read> src_radius_word: array<u32>;
@group(0) @binding(9) var<storage, read_write> dst_rows: array<atomic<u32>>;

const CROWN_COEFF_MAX: f32 = 1e10;
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
// up_f32(u/(1-u)), u=2^-24; exact f32 bits 0x33800001.
const GAMMA_1_UP: f32 = 5.9604652e-8;

fn nonfinite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}
fn center_bad(v: f32) -> bool {
    return nonfinite(v) || abs(v) >= CROWN_COEFF_MAX;
}
fn radius_bad(v: f32) -> bool {
    return nonfinite(v) || v < 0.0 || v >= CROWN_COEFF_MAX;
}
fn round_up_pos(v: f32) -> f32 {
    let bits = bitcast<u32>(v);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return v; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.stride == 0u || p.dim == 0u) { return; }
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        let a = dst_center[i];
        let b = src_center[i];
        let ea = dst_radius[i];
        let eb = src_radius[i];
        let s = a + b;
        let rho = round_up_pos(abs(s) * GAMMA_1_UP);
        let radius_sum = round_up_pos(ea + eb);
        let e = round_up_pos(radius_sum + rho);

        var center_word = dst_center_word[i] | src_center_word[i];
        var radius_word = dst_radius_word[i] | src_radius_word[i];
        if (center_bad(a) || center_bad(b) || center_bad(s)) { center_word |= 1u; }
        if (radius_bad(ea) || radius_bad(eb) || radius_bad(e)) { radius_word |= 1u; }
        dst_center[i] = s;
        dst_radius[i] = e;
        dst_center_word[i] = center_word;
        dst_radius_word[i] = radius_word;
        atomicOr(&dst_rows[i / p.dim], center_word | radius_word);
        i += p.stride;
    }
}
"#;

/// Merge both bias centre/radius pairs and OR the source row accumulator. Bias
/// has no separate per-entry word buffer, so any invalid input or result is born
/// directly into the destination row word. One invocation owns each row; the
/// atomic destination remains compatible with the resident walk's row-OR type.
pub(in crate::wgpu_device) const SWEEP_MERGE_BIAS_ROWS_SHADER: &str = r#"
struct Params { n: u32, stride: u32, dim: u32, _padding: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> dst_lower_bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_lower_radius: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_upper_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_upper_radius: array<f32>;
@group(0) @binding(5) var<storage, read> src_lower_bias: array<f32>;
@group(0) @binding(6) var<storage, read> src_lower_radius: array<f32>;
@group(0) @binding(7) var<storage, read> src_upper_bias: array<f32>;
@group(0) @binding(8) var<storage, read> src_upper_radius: array<f32>;
@group(0) @binding(9) var<storage, read_write> dst_rows: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> src_rows: array<u32>;

const CROWN_COEFF_MAX: f32 = 1e10;
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
const GAMMA_1_UP: f32 = 5.9604652e-8;

fn nonfinite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}
fn center_bad(v: f32) -> bool {
    return nonfinite(v) || abs(v) >= CROWN_COEFF_MAX;
}
fn radius_bad(v: f32) -> bool {
    return nonfinite(v) || v < 0.0 || v >= CROWN_COEFF_MAX;
}
fn round_up_pos(v: f32) -> f32 {
    let bits = bitcast<u32>(v);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return v; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.stride == 0u) { return; }
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        let al = dst_lower_bias[i];
        let bl = src_lower_bias[i];
        let eal = dst_lower_radius[i];
        let ebl = src_lower_radius[i];
        let sl = al + bl;
        let rhol = round_up_pos(abs(sl) * GAMMA_1_UP);
        let el = round_up_pos(round_up_pos(eal + ebl) + rhol);

        let au = dst_upper_bias[i];
        let bu = src_upper_bias[i];
        let eau = dst_upper_radius[i];
        let ebu = src_upper_radius[i];
        let su = au + bu;
        let rhou = round_up_pos(abs(su) * GAMMA_1_UP);
        let eu = round_up_pos(round_up_pos(eau + ebu) + rhou);

        var word = src_rows[i];
        if (center_bad(al) || center_bad(bl) || center_bad(sl) ||
            center_bad(au) || center_bad(bu) || center_bad(su) ||
            radius_bad(eal) || radius_bad(ebl) || radius_bad(el) ||
            radius_bad(eau) || radius_bad(ebu) || radius_bad(eu)) {
            word |= 1u;
        }
        dst_lower_bias[i] = sl;
        dst_lower_radius[i] = el;
        dst_upper_bias[i] = su;
        dst_upper_radius[i] = eu;
        atomicOr(&dst_rows[i], word);
        i += p.stride;
    }
}
"#;

/// Fold the four final per-entry word lanes into the sticky per-row
/// accumulator before the one transaction-final readback. Keeping this as one
/// dispatch avoids four temporary row buffers and makes the carrier download
/// working set exact.
pub(in crate::wgpu_device) const SWEEP_FINALIZE_ROWS_SHADER: &str = r#"
struct Params { n: u32, stride: u32, dim: u32, _padding: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> lower_center_word: array<u32>;
@group(0) @binding(2) var<storage, read> upper_center_word: array<u32>;
@group(0) @binding(3) var<storage, read> lower_radius_word: array<u32>;
@group(0) @binding(4) var<storage, read> upper_radius_word: array<u32>;
@group(0) @binding(5) var<storage, read_write> rows: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.stride == 0u || p.dim == 0u) { return; }
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        let word = lower_center_word[i] | upper_center_word[i] |
                   lower_radius_word[i] | upper_radius_word[i];
        atomicOr(&rows[i / p.dim], word);
        i += p.stride;
    }
}
"#;

/// Pipeline plus the exact bind-group layout its encoder must use.
pub(in crate::wgpu_device) struct SweepDagPipeline {
    pub(in crate::wgpu_device) pipeline: wgpu::ComputePipeline,
    pub(in crate::wgpu_device) layout: wgpu::BindGroupLayout,
}

/// Per-device cached pipeline bundle. Full verdict qualification materializes
/// it before authority is published; pipeline/compiler residency is opaque and
/// outside the logical numerical-buffer receipt.
pub(in crate::wgpu_device) struct SweepDagPipelines {
    pub(in crate::wgpu_device) inject_values: SweepDagPipeline,
    pub(in crate::wgpu_device) inject_words: SweepDagPipeline,
    pub(in crate::wgpu_device) negate_centers: SweepDagPipeline,
    pub(in crate::wgpu_device) merge_matrix: SweepDagPipeline,
    pub(in crate::wgpu_device) merge_bias_rows: SweepDagPipeline,
    pub(in crate::wgpu_device) finalize_rows: SweepDagPipeline,
}

impl SweepDagPipelines {
    /// Build every module through the device's already-resolved shader-loading
    /// profile. The caller must hold full-IEEE authority before calling this.
    pub(in crate::wgpu_device) fn create(device: &wgpu::Device, denorm_preserve: bool) -> Self {
        Self {
            inject_values: create_pipeline(
                device,
                denorm_preserve,
                "sweep_inject_values",
                SWEEP_INJECT_VALUES_SHADER,
                &[true, false, false, false, false, false, false, false, false],
            ),
            inject_words: create_pipeline(
                device,
                denorm_preserve,
                "sweep_inject_words",
                SWEEP_INJECT_WORDS_SHADER,
                &[true, false, false, false, false, false],
            ),
            negate_centers: create_pipeline(
                device,
                denorm_preserve,
                "sweep_negate_centers",
                SWEEP_NEGATE_CENTERS_SHADER,
                &[false, false],
            ),
            merge_matrix: create_pipeline(
                device,
                denorm_preserve,
                "sweep_merge_matrix",
                SWEEP_MERGE_MATRIX_SHADER,
                &[false, false, false, false, true, true, true, true, false],
            ),
            merge_bias_rows: create_pipeline(
                device,
                denorm_preserve,
                "sweep_merge_bias_rows",
                SWEEP_MERGE_BIAS_ROWS_SHADER,
                &[
                    false, false, false, false, true, true, true, true, false, true,
                ],
            ),
            finalize_rows: create_pipeline(
                device,
                denorm_preserve,
                "sweep_finalize_rows",
                SWEEP_FINALIZE_ROWS_SHADER,
                &[true, true, true, true, false],
            ),
        }
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    denorm_preserve: bool,
    label: &str,
    source: &str,
    storage_read_only: &[bool],
) -> SweepDagPipeline {
    let module = crate::wgpu_device::shader_loading::create_compute_module(
        device,
        denorm_preserve,
        &format!("{label}_shader"),
        source,
    );
    let mut entries = Vec::with_capacity(storage_read_only.len() + 1);
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    for (index, &read_only) in storage_read_only.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: u32::try_from(index + 1).expect("sweep binding count is static and tiny"),
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bind_group_layout")),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pipeline_layout")),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    SweepDagPipeline { pipeline, layout }
}

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;

    const GAMMA_1_UP: f32 = f32::from_bits(0x3380_0001);

    fn round_up_pos(value: f32) -> f32 {
        let bits = value.to_bits();
        let magnitude = bits & 0x7fff_ffff;
        if magnitude >= 0x7f80_0000 {
            return value;
        }
        if bits & 0x8000_0000 != 0 || magnitude == 0 {
            return 0.0;
        }
        if magnitude < 0x0080_0000 {
            return f32::MIN_POSITIVE;
        }
        f32::from_bits(bits + 1)
    }

    fn seed_center_word(value: f32) -> u32 {
        u32::from(!value.is_finite() || value.abs() >= ny_core::CROWN_COEFF_MAX)
    }

    fn seed_radius_word(value: f32) -> u32 {
        u32::from(!(0.0..ny_core::CROWN_COEFF_MAX).contains(&value))
    }

    fn merge_model(a: f32, ea: f32, aw: u32, b: f32, eb: f32, bw: u32) -> (f32, f32, u32, u32) {
        let s = a + b;
        let rho = round_up_pos(s.abs() * GAMMA_1_UP);
        let e = round_up_pos(round_up_pos(ea + eb) + rho);
        (
            s,
            e,
            aw | bw | seed_center_word(a) | seed_center_word(b) | seed_center_word(s),
            seed_radius_word(ea) | seed_radius_word(eb) | seed_radius_word(e),
        )
    }

    #[test]
    fn shaders_parse_validate_and_stay_within_the_binding_budget() {
        for (name, source, storage_bindings) in [
            ("inject values", SWEEP_INJECT_VALUES_SHADER, 9usize),
            ("inject words", SWEEP_INJECT_WORDS_SHADER, 6),
            ("negate", SWEEP_NEGATE_CENTERS_SHADER, 2),
            ("merge matrix", SWEEP_MERGE_MATRIX_SHADER, 9),
            ("merge bias", SWEEP_MERGE_BIAS_ROWS_SHADER, 10),
            ("finalize rows", SWEEP_FINALIZE_ROWS_SHADER, 5),
        ] {
            assert!(
                storage_bindings <= 11,
                "{name} exceeds the audited binding budget"
            );
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} WGSL parse failed: {error:?}"));
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} WGSL validation failed: {error:?}"));
        }
    }

    #[test]
    fn merge_model_encloses_the_exact_f64_sum_and_both_input_radii() {
        let edge = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MAX.min(9.0e9),
            -f32::MAX.min(9.0e9),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
            1.0 + f32::EPSILON,
            16_777_216.0,
        ];
        for &a in &edge {
            for &b in &edge {
                for &(ea, eb) in &[(0.0, 0.0), (f32::from_bits(1), 0.25), (1.0, 2.0)] {
                    assert_merge_encloses(a, ea, b, eb);
                }
            }
        }

        // Deterministic broad bit-pattern hunt without adding a rand dependency.
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..200_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = f32::from_bits(state);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = f32::from_bits(state);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let ea = f32::from_bits(state & 0x3fff_ffff);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let eb = f32::from_bits(state & 0x3fff_ffff);
            if a.is_finite()
                && b.is_finite()
                && a.abs() < ny_core::CROWN_COEFF_MAX / 4.0
                && b.abs() < ny_core::CROWN_COEFF_MAX / 4.0
            {
                assert_merge_encloses(a, ea, b, eb);
            }
        }
    }

    fn assert_merge_encloses(a: f32, ea: f32, b: f32, eb: f32) {
        let (s, e, center_word, radius_word) = merge_model(a, ea, 0, b, eb, 0);
        if center_word != 0 || radius_word != 0 {
            return;
        }
        let exact_sum = f64::from(a) + f64::from(b);
        let exact_gap = (f64::from(s) - exact_sum).abs();
        let required = f64::from(ea) + f64::from(eb) + exact_gap;
        assert!(
            f64::from(e) >= required,
            "merge under-covered: a={a:e} b={b:e} ea={ea:e} eb={eb:e} \
             s={s:e} e={e:e} required={required:e}"
        );
    }

    #[test]
    fn cancellation_cannot_launder_a_branch_word() {
        let (center, _, center_word, _) = merge_model(1.0, 0.0, 0, -1.0, 0.0, 1);
        assert_eq!(center, 0.0);
        assert_eq!(center_word, 1);
    }

    #[test]
    fn binary_bias_token_is_counted_once_but_branch_biases_both_survive() {
        let incoming_bias = 3.0f32;
        let lhs_branch_bias = 5.0f32;
        let rhs_branch_bias = 7.0f32;
        let lhs = incoming_bias + lhs_branch_bias;
        let rhs_biasless_fork = 0.0 + rhs_branch_bias;
        let (merged, _, word, _) = merge_model(lhs, 0.0, 0, rhs_biasless_fork, 0.0, 0);
        assert_eq!(merged, 15.0);
        assert_eq!(word, 0);
        assert_ne!(merged, 18.0, "incoming bias was duplicated across the fork");
    }

    #[test]
    fn sub_negates_each_side_without_swapping_lower_and_upper() {
        let lower = 2.0f32;
        let upper = -7.0f32;
        let negated_lower = f32::from_bits(lower.to_bits() ^ 0x8000_0000);
        let negated_upper = f32::from_bits(upper.to_bits() ^ 0x8000_0000);
        assert_eq!((negated_lower, negated_upper), (-2.0, 7.0));
        assert_ne!((negated_lower, negated_upper), (7.0, -2.0));
        let compact: String = SWEEP_NEGATE_CENTERS_SHADER
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        assert!(compact
            .contains("lower_center[i]=bitcast<f32>(bitcast<u32>(lower_center[i])^0x80000000u)"));
        assert!(compact
            .contains("upper_center[i]=bitcast<f32>(bitcast<u32>(upper_center[i])^0x80000000u)"));
    }

    #[test]
    fn shader_gamma_literal_is_the_outward_host_constant() {
        let u = 2.0f64.powi(-24);
        let exact = u / (1.0 - u);
        assert_eq!(GAMMA_1_UP.to_bits(), 0x3380_0001);
        assert!(f64::from(GAMMA_1_UP) >= exact);
        assert!(SWEEP_MERGE_MATRIX_SHADER.contains("0x33800001"));
    }
}
