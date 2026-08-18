// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #rung3-denorm-chase: characterize the GB10's plain and DenormPreserve
//! subnormal behavior, including the direct-FMA form production now avoids.
//!
//! MEASURED OUTCOME (2026-08-10, GB10/Vulkan): plain WGSL fails the gradual-
//! underflow policy. DenormPreserve makes core add/multiply preserve the signed
//! subnormal endpoint lanes, but direct `fma(subnormal, large, 0)` still
//! DAZ-zeroes its multiplicand even when the expected result is normal.
//! Production EFT primary products consequently use the qualified core
//! multiply; FMA remains only in residual forms that conservatively over-charge
//! that DAZ behavior or are covered by the scaled rung-3 floor. With the armed
//! production taint-word channel, the live GB10/DenormPreserve ladder now
//! measures 5/5. U5/U6 and B0 are discharged and the raw CROWN source gate is
//! open; an exact process request and this passing ladder are still required,
//! while public proof use still requires the typed per-device five-rung
//! qualification constructor.
//!
//! THE MECHANISM TESTED. The pinned naga 29.0.3 SPIR-V backend emits NO
//! float-control execution modes (verified
//! by grep: zero mentions of DenormPreserve in `back/spv/`). So the flush may
//! be a TOOLCHAIN GAP: inject `OpCapability DenormPreserve` +
//! `OpExtension "SPV_KHR_float_controls"` + `OpExecutionMode %main
//! DenormPreserve 32` into naga's output and load it through wgpu's
//! `PASSTHROUGH_SHADERS` path, and the same probe kernel may stop flushing.
//!
//! This executable is a REPORTING probe, not a libtest gate: run it explicitly
//! with `cargo run -p ny-gpu --features gpu-tests --example
//! denorm-preserve-probe`. It prints what it finds and
//! asserts only its own internal consistency. Whatever it reports, the live
//! authority ladder's narrower rung 3 remains the sole authority input. This
//! diagnostic cannot open the compensated channel or verdict authority.
//!
//! Fail-closed discipline: every failure (missing adapter feature, naga
//! compile error, driver rejecting the injected module or pipeline) is
//! REPORTED as `flush-preserved = unknown (refused)`, never as success.

#![cfg(feature = "gpu-tests")]
// Integration test (not a lib module) because the passthrough API is `unsafe`
// and the lib rightly carries `#![forbid(unsafe_code)]`. The one unsafe call is
// scoped and annotated below.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

/// Superset of `wgpu_device/ops/subnormal_selfcheck.rs`'s authority operations,
/// plus historical bitcast-TwoSum candidates and direct-FMA characterization.
/// Duplicated (not exported) so the lib keeps its gate private. The add,
/// multiply, and barrier-residual lanes mirror the live gate; direct-FMA lanes
/// deliberately do not, because production excludes that measured-bad form.
const SUBNORMAL_SELFCHECK_SHADER: &str = r#"
struct Params { n_pairs: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       inp:  array<u32>;
@group(0) @binding(2) var<storage, read_write>  outp: array<u32>;

// fma-barrier TwoSum: byte-identical to eft_selfcheck.rs / the shipped kernels.
fn two_sum_fma_barrier(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = fma(-1.0, a, s);   // s - a
    let sb = fma(-1.0, bb, s);  // s - bb
    let da = fma(-1.0, sb, a);  // a - (s - bb)
    let db = fma(-1.0, bb, b);  // b - bb
    return vec2<f32>(s, da + db);
}

// CANDIDATE #rung3-fma-dodge: Knuth TwoSum with a UNIFORM-SOURCED bitcast
// barrier. MEASURED (previous revision of this probe): a pure bitcast
// round-trip (^ 0u literal) is constant-folded away and the residual algebra
// reassociates to exactly 0 on every pair — the July "plain Knuth/Dekker
// destroyed" result. Here the barrier xors in `params.pad1`, a runtime uniform
// the compiler cannot prove zero, so the bitcast chain and its ordering must
// survive. Every op stays core add/sub — which, under DenormPreserve, the GB10
// preserves — so the residual dodges the flushing fma lane entirely.
fn bar(x: f32, z: u32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ z); }
fn two_sum_bitcast_barrier(a: f32, b: f32, z: u32) -> vec2<f32> {
    let s = bar(a + b, z);
    let bb = bar(s - a, z);
    let da = a - bar(s - bb, z);
    let db = b - bb;
    return vec2<f32>(s, bar(da + db, z));
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        let a = bitcast<f32>(inp[2u * i]);
        let b = bitcast<f32>(inp[2u * i + 1u]);
        let ts = two_sum_fma_barrier(a, b);
        let tb = two_sum_bitcast_barrier(a, b, params.pad1);
        outp[7u * i]      = bitcast<u32>(a + b);
        outp[7u * i + 1u] = bitcast<u32>(a * b);
        outp[7u * i + 2u] = bitcast<u32>(ts.y);
        // fma lanes: addend passthrough (a*0 + b) and true fused product+add.
        outp[7u * i + 3u] = bitcast<u32>(fma(a, 0.0, b));
        outp[7u * i + 4u] = bitcast<u32>(fma(a, b, 0.0));
        // bitcast-barrier TwoSum: sum and residual.
        outp[7u * i + 5u] = bitcast<u32>(tb.x);
        outp[7u * i + 6u] = bitcast<u32>(tb.y);
    }
}
"#;

/// SPIR-V opcodes and enum values (SPIR-V 1.x spec).
const OP_CAPABILITY: u32 = 17;
const OP_EXTENSION: u32 = 10;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const CAP_DENORM_PRESERVE: u32 = 4464;
const MODE_DENORM_PRESERVE: u32 = 4459;

fn word(op: u32, wc: u32) -> u32 {
    (wc << 16) | op
}

/// Encode a NUL-terminated string literal as SPIR-V words.
fn string_words(s: &str) -> Vec<u32> {
    let mut bytes: Vec<u8> = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Inject DenormPreserve into a naga-emitted SPIR-V module.
///
/// Returns `None` (fail-closed) if the module does not have the expected
/// section structure. Section order per spec: capabilities, extensions,
/// ext-inst imports, memory model, entry points, execution modes, ...
fn inject_denorm_preserve(spirv: &[u32]) -> Option<Vec<u32>> {
    if spirv.len() < 6 || spirv[0] != 0x0723_0203 {
        return None; // not a SPIR-V module we understand
    }
    let mut out: Vec<u32> = spirv[..5].to_vec();
    let mut i = 5usize;
    let mut last_cap_end = None;
    let mut entry_point_id = None;
    let mut exec_mode_insert_at = None;

    // First pass over the copied stream: find insertion points.
    while i < spirv.len() {
        let instr = spirv[i];
        let opcode = instr & 0xFFFF;
        let wc = (instr >> 16) as usize;
        if wc == 0 || i + wc > spirv.len() {
            return None; // malformed
        }
        if opcode == OP_CAPABILITY {
            last_cap_end = Some(i + wc);
        }
        if opcode == OP_ENTRY_POINT && entry_point_id.is_none() {
            // OpEntryPoint ExecutionModel <id> "name" interfaces...
            entry_point_id = Some(spirv[i + 2]);
        }
        if (opcode == OP_EXECUTION_MODE || opcode == OP_ENTRY_POINT) && i + wc <= spirv.len() {
            exec_mode_insert_at = Some(i + wc);
        }
        i += wc;
    }
    let cap_end = last_cap_end?;
    let entry = entry_point_id?;
    let mode_at = exec_mode_insert_at?;

    // Build the module with injections, preserving order.
    out.clear();
    out.extend_from_slice(&spirv[..cap_end]);
    // OpCapability DenormPreserve
    out.push(word(OP_CAPABILITY, 2));
    out.push(CAP_DENORM_PRESERVE);
    // OpExtension "SPV_KHR_float_controls"
    let ext = string_words("SPV_KHR_float_controls");
    out.push(word(OP_EXTENSION, 1 + ext.len() as u32));
    out.extend_from_slice(&ext);
    // Copy up to the execution-mode insertion point, then inject the mode.
    out.extend_from_slice(&spirv[cap_end..mode_at]);
    out.push(word(OP_EXECUTION_MODE, 4));
    out.push(entry);
    out.push(MODE_DENORM_PRESERVE);
    out.push(32); // bit width
    out.extend_from_slice(&spirv[mode_at..]);
    // Sanity: extension/ext-inst/memory-model opcodes only appear where legal.
    Some(out)
}

/// Compile WGSL to SPIR-V via the same naga the shipped pipeline uses.
fn wgsl_to_spirv(wgsl: &str) -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("wgsl parse: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("validate: {e:?}"))?;
    let options = naga::back::spv::Options::default();
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .map_err(|e| format!("spv write: {e}"))
}

/// The decisive operand pairs (bit patterns), mirroring the ladder probe's
/// discrimination:
///   pair 0: a = min subnormal (2^-149), b = 0.0
///           a+b lane: DAZ reads a as 0 -> 0x00000000; honest -> 0x00000001
///   pair 1: a = 2^-74, b = 2^-75
///           a*b lane: product 2^-149 exactly; FTZ writes 0; honest 0x00000001
const PAIRS: [(u32, u32); 10] = [
    (0x0000_0001, 0x0000_0000), // (2^-149, 0.0)
    (0x1A80_0000, 0x1A00_0000), // (2^-74, 2^-75): product = 2^-149 exactly
    (0x3F80_0000, 0x0000_0001), // (1.0, 2^-149): subnormal TwoSum residual
    // Adversarial EXACTNESS pairs (normal-range): the residual is nonzero and
    // exactly representable; an unbarriered TwoSum reassociates these to 0.
    (0x4CBE_BC20, 0x3F80_0000), // (1e8, 1.0): s=1e8, residual must be exactly 1.0
    (0x3F80_0000, 0x3400_0001), // (1.0, 2^-23·(1+2^-23)): sub-ulp tail
    (0x7149_F2CA, 0x0D00_0000), // (1e30, 2^-101·...): extreme exponent gap -> residual = b
    // Diagnostic-only direct-FMA operand-DAZ discriminators: expected results
    // are normal, but this form is deliberately excluded from the live gate.
    (0x007F_FFFF, 0x4E80_0000), // maxsub * 2^30 = 0x0f7ffffe
    (0x007F_FFFF, 0xCE80_0000), // maxsub * -2^30 = 0x8f7ffffe
    (0x0000_0001, 0x4E80_0000), // minsub * 2^30 = 0x04000000
    (0x0000_0001, 0xCE80_0000), // minsub * -2^30 = 0x84000000
];

struct LaneReport {
    add_preserved: bool,
    mul_preserved: bool,
    /// fma(x, 0.0, s) where s = 2^-149: the addend passes through fused.
    fma_addend_preserved: bool,
    /// fma(a, b, 0.0) where a*b = 2^-149 exactly: fused product is subnormal.
    fma_product_preserved: bool,
    /// Direct FMA must not DAZ a subnormal multiplicand when amplification makes
    /// the mathematically expected result normal (both signs are checked).
    fma_operand_preserved: bool,
    /// bitcast-barrier TwoSum: bit-exact vs the CPU reference on EVERY pair
    /// (sum and residual) — the exactness question (does the barrier hold?).
    bitcast_twosum_exact: bool,
    /// bitcast-barrier TwoSum residual of (1.0, 2^-149) == 2^-149 — the
    /// subnormal question (does it dodge the fma flush?).
    bitcast_twosum_subnormal: bool,
}

/// CPU TwoSum (Knuth), plain f32 — Rust scalar arithmetic is IEEE with gradual
/// underflow, so this is the ground truth for both questions.
fn cpu_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

fn run_probe_with_module(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
) -> Result<LaneReport, String> {
    let mut input: Vec<u32> = Vec::new();
    for (a, b) in PAIRS {
        input.push(a);
        input.push(b);
    }
    let n_pairs = PAIRS.len() as u32;
    let params: [u32; 4] = [n_pairs, 0, 0, 0];

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("denorm-probe-params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("denorm-probe-in"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (7 * PAIRS.len() * 4) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("denorm-probe-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("denorm-probe-read"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Passthrough modules carry no reflection, so the layout must be explicit.
    // Both arms use this same layout, keeping the comparison exact.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("denorm-probe-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("denorm-probe-pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("denorm-probe-pipeline"),
        layout: Some(&pl),
        module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("denorm-probe-bind"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
    queue.submit([enc.finish()]);

    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .map_err(|_| "map channel closed".to_string())?
        .map_err(|e| format!("map failed: {e:?}"))?;
    let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    read_buf.unmap();

    // Layout: 7 lanes per pair — [a+b, a*b, fma_twosum.res, fma(a,0,b),
    // fma(a,b,0), bitcast_twosum.sum, bitcast_twosum.res].
    let mut bitcast_exact = true;
    for (i, (a_bits, b_bits)) in PAIRS.iter().enumerate() {
        let (a, b) = (f32::from_bits(*a_bits), f32::from_bits(*b_bits));
        let (cs, cr) = cpu_two_sum(a, b);
        let gs = data[7 * i + 5];
        let gr = data[7 * i + 6];
        if gs != cs.to_bits() || gr != cr.to_bits() {
            println!(
                "  bitcast-twosum MISMATCH pair {i}: gpu=(0x{gs:08x},0x{gr:08x})                  cpu=(0x{:08x},0x{:08x})",
                cs.to_bits(),
                cr.to_bits()
            );
            bitcast_exact = false;
        }
    }
    Ok(LaneReport {
        add_preserved: data[0] == 0x0000_0001,
        mul_preserved: data[7 + 1] == 0x0000_0001,
        fma_addend_preserved: data[2 * 7 + 3] == 0x0000_0001,
        fma_product_preserved: data[7 + 4] == 0x0000_0001,
        fma_operand_preserved: data[6 * 7 + 4] == 0x0F7F_FFFE
            && data[7 * 7 + 4] == 0x8F7F_FFFE
            && data[8 * 7 + 4] == 0x0400_0000
            && data[9 * 7 + 4] == 0x8400_0000,
        bitcast_twosum_exact: bitcast_exact,
        bitcast_twosum_subnormal: data[2 * 7 + 6] == 0x0000_0001,
    })
}

/// The chase, end to end. Prints a rung-style report; never opens anything.
fn main() {
    let instance = wgpu::Instance::default();
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("denorm-chase: no adapter ({e:?}); selected probe cannot run");
            std::process::exit(2);
        }
    };
    let features = adapter.features();
    let has_passthrough = features.contains(wgpu::Features::PASSTHROUGH_SHADERS);
    println!(
        "denorm-chase: adapter = {} / {:?}; PASSTHROUGH_SHADERS = {has_passthrough}",
        adapter.get_info().name,
        adapter.get_info().backend
    );

    let mut requested = wgpu::Features::empty();
    if has_passthrough {
        requested |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("denorm-chase"),
            required_features: requested,
            ..Default::default()
        })) {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("denorm-chase: device creation failed ({e}); selected probe cannot run");
                std::process::exit(2);
            }
        };

    let wgsl = SUBNORMAL_SELFCHECK_SHADER;

    // Arm 1: the ordinary WGSL path — expected to FLUSH on this box.
    let wgsl_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("denorm-probe-wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });
    match run_probe_with_module(&device, &queue, &wgsl_module) {
            Ok(r) => println!(
                "denorm-chase: WGSL path        add={} mul={} fma_addend={} fma_product={} fma_operand={}                  bitcast_twosum_exact={} bitcast_twosum_subnormal={}",
                r.add_preserved, r.mul_preserved, r.fma_addend_preserved,
                r.fma_product_preserved, r.fma_operand_preserved, r.bitcast_twosum_exact,
                r.bitcast_twosum_subnormal
            ),
            Err(e) => {
                eprintln!("denorm-chase: WGSL path refused: {e}");
                std::process::exit(1);
            }
        }

    // Arm 2: naga SPIR-V + DenormPreserve injection via passthrough.
    if !has_passthrough {
        eprintln!(
            "denorm-chase: PASSTHROUGH_SHADERS unavailable; injected arm = \
                 unknown (refused). The chase needs a different loading path."
        );
        std::process::exit(2);
    }
    let spirv = match wgsl_to_spirv(wgsl) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("denorm-chase: naga spv compile refused: {e}");
            std::process::exit(1);
        }
    };
    let injected = match inject_denorm_preserve(&spirv) {
        Some(w) => w,
        None => {
            eprintln!("denorm-chase: injection refused (unexpected module shape)");
            std::process::exit(1);
        }
    };
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    #[allow(unsafe_code)]
    let spv_module = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("denorm-probe-injected"),
            spirv: Some(Cow::Owned(injected)),
            ..Default::default()
        })
    };
    if let Some(e) = pollster::block_on(validation_scope.pop()) {
        eprintln!("denorm-chase: driver rejected injected module: {e}");
        std::process::exit(1);
    }
    match run_probe_with_module(&device, &queue, &spv_module) {
        Ok(r) => {
            println!(
                    "denorm-chase: INJECTED path    add={} mul={} fma_addend={} fma_product={} fma_operand={}                      bitcast_twosum_exact={} bitcast_twosum_subnormal={}",
                    r.add_preserved, r.mul_preserved, r.fma_addend_preserved,
                    r.fma_product_preserved, r.fma_operand_preserved,
                    r.bitcast_twosum_exact, r.bitcast_twosum_subnormal
                );
            if r.add_preserved && r.mul_preserved && r.fma_operand_preserved {
                println!(
                    "denorm-chase: *** DenormPreserve injection WORKS on this \
                         adapter for core arithmetic and direct FMA operands. ***"
                );
            } else if r.add_preserved && r.mul_preserved {
                println!(
                    "denorm-chase: core add/mul preservation WORKS, but direct \
                     fma(subnormal,large,0) still DAZ-zeroes its operand. The \
                     production EFT writers therefore use core mul for primary \
                     products; FMA remains only in conservative/floor-covered \
                     residual forms. Re-run the live rung-3 gate, not this raw \
                     direct-FMA diagnostic, for authorization."
                );
            } else {
                println!(
                    "denorm-chase: injection accepted but flush persists — the \
                         driver ignores DenormPreserve here; rung 3 needs the \
                         certified subnormal-floor route instead."
                );
            }
        }
        Err(e) => {
            eprintln!("denorm-chase: injected path refused: {e}");
            std::process::exit(1);
        }
    }
}
