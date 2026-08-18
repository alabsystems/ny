// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #rung3-denorm-chase, production arm: compile WGSL through the SAME naga the
//! WGSL path uses, inject the three SPIR-V float-control instructions that make
//! the driver preserve subnormals, and load the result through wgpu's
//! passthrough API.
//!
//! MEASURED BASIS (2026-08-10, NVIDIA GB10/Vulkan,
//! `ny-gpu/tests/denorm_preserve_probe.rs`): the identical probe kernel flushes
//! subnormals via the plain WGSL path (`add_preserved=false
//! mul_preserved=false`) and preserves them bit-exactly with the injection
//! (`add_preserved=true mul_preserved=true`). naga 29.0.3 emits no
//! float-control execution modes at all, so without this the NVIDIA driver
//! defaults to flush-to-zero — which fails authority-ladder rung 3 and drags
//! rung 2 down via the #u2b entailment.
//!
//! # Safety posture
//!
//! `create_shader_module_passthrough` is `unsafe` because wgpu performs no
//! validation on the words. This crate narrows that to a checked contract:
//!
//! * the words are naga's OWN output for the given WGSL (same front/back ends
//!   as the safe path), post-processed only by inserting three well-formed
//!   instructions whose section placement follows the SPIR-V logical layout;
//! * the call runs under a `Validation` error scope, and any driver rejection
//!   returns `Err` — the caller falls back to the safe WGSL path;
//! * nothing here is a verdict claim. Authority remains governed by
//!   `ny-gpu`'s `sound_authority.rs` ladder, which MEASURES the result.
//!
//! The injector is pure and CPU-tested below; the unsafe call is 6 lines.

use std::borrow::Cow;

/// SPIR-V opcodes and enum values (SPIR-V 1.x spec).
const OP_CAPABILITY: u32 = 17;
const OP_EXTENSION: u32 = 10;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const CAP_DENORM_PRESERVE: u32 = 4464;
const MODE_DENORM_PRESERVE: u32 = 4459;
const SPIRV_MAGIC: u32 = 0x0723_0203;

fn word(op: u32, wc: u32) -> u32 {
    (wc << 16) | op
}

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

/// Inject `DenormPreserve` (32-bit) into a SPIR-V module: one capability, one
/// extension, and one execution mode PER ENTRY POINT. Returns `None`
/// (fail-closed) on any module we do not fully understand.
pub fn inject_denorm_preserve(spirv: &[u32]) -> Option<Vec<u32>> {
    if spirv.len() < 6 || spirv[0] != SPIRV_MAGIC {
        return None;
    }
    let mut i = 5usize;
    let mut last_cap_end = None;
    let mut entry_point_ids: Vec<u32> = Vec::new();
    let mut mode_insert_at = None;
    while i < spirv.len() {
        let instr = spirv[i];
        let opcode = instr & 0xFFFF;
        let wc = (instr >> 16) as usize;
        if wc == 0 || i + wc > spirv.len() {
            return None;
        }
        if opcode == OP_CAPABILITY {
            last_cap_end = Some(i + wc);
        }
        if opcode == OP_ENTRY_POINT {
            entry_point_ids.push(spirv[i + 2]);
        }
        if opcode == OP_ENTRY_POINT || opcode == OP_EXECUTION_MODE {
            mode_insert_at = Some(i + wc);
        }
        i += wc;
    }
    let cap_end = last_cap_end?;
    let mode_at = mode_insert_at?;
    if entry_point_ids.is_empty() {
        return None;
    }
    let mut out: Vec<u32> = Vec::with_capacity(spirv.len() + 16);
    out.extend_from_slice(&spirv[..cap_end]);
    out.push(word(OP_CAPABILITY, 2));
    out.push(CAP_DENORM_PRESERVE);
    let ext = string_words("SPV_KHR_float_controls");
    out.push(word(OP_EXTENSION, 1 + ext.len() as u32));
    out.extend_from_slice(&ext);
    out.extend_from_slice(&spirv[cap_end..mode_at]);
    for entry in &entry_point_ids {
        out.push(word(OP_EXECUTION_MODE, 4));
        out.push(*entry);
        out.push(MODE_DENORM_PRESERVE);
        out.push(32);
    }
    out.extend_from_slice(&spirv[mode_at..]);
    Some(out)
}

/// Compile WGSL to SPIR-V with the same naga the WGSL path uses. Compiles the
/// WHOLE module (no per-entry-point pipeline options) so entry-point selection
/// at pipeline creation behaves exactly like the WGSL path.
pub fn wgsl_to_spirv(wgsl: &str) -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("wgsl parse: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("validate: {e:?}"))?;
    let options = naga::back::spv::Options::default();
    naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("spv write: {e}"))
}

/// Compile `wgsl` into a denorm-preserving shader module via passthrough.
///
/// Fail-closed: every failure (compile, injection, driver validation) returns
/// `Err(reason)` and creates nothing the caller must clean up. The caller is
/// expected to fall back to `device.create_shader_module` with the same WGSL.
pub fn create_denorm_preserving_module(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
) -> Result<wgpu::ShaderModule, String> {
    if !device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        return Err("PASSTHROUGH_SHADERS not enabled on this device".into());
    }
    let spirv = wgsl_to_spirv(wgsl)?;
    let injected = inject_denorm_preserve(&spirv)
        .ok_or_else(|| "injection refused (module shape)".to_string())?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // SAFETY: `injected` is naga's own output for this WGSL plus three
    // well-formed instructions placed per the SPIR-V logical layout; the
    // validation scope below converts any driver rejection into `Err`.
    #[allow(unsafe_code)]
    let module = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some(label),
            spirv: Some(Cow::Owned(injected)),
            ..Default::default()
        })
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(format!("driver rejected injected module: {e}"));
    }
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2.0;
}
"#;

    /// CPU-only: the injector inserts exactly one capability, one extension,
    /// and one execution mode per entry point, and the module still parses as
    /// a well-formed instruction stream.
    #[test]
    fn injector_inserts_wellformed_instructions() {
        let spirv = wgsl_to_spirv(TINY).expect("naga compiles the tiny shader");
        let injected = inject_denorm_preserve(&spirv).expect("injection succeeds");
        assert!(injected.len() > spirv.len());
        // Walk the injected stream: must remain well-formed, and must contain
        // our three additions.
        let (mut caps, mut exts, mut modes) = (0, 0, 0);
        let mut i = 5;
        while i < injected.len() {
            let opcode = injected[i] & 0xFFFF;
            let wc = (injected[i] >> 16) as usize;
            assert!(wc > 0 && i + wc <= injected.len(), "malformed at {i}");
            match opcode {
                OP_CAPABILITY if injected[i + 1] == CAP_DENORM_PRESERVE => caps += 1,
                OP_EXTENSION => exts += 1,
                OP_EXECUTION_MODE if injected[i + 2] == MODE_DENORM_PRESERVE => {
                    assert_eq!(injected[i + 3], 32);
                    modes += 1;
                }
                _ => {}
            }
            i += wc;
        }
        assert_eq!(caps, 1, "exactly one DenormPreserve capability");
        assert!(exts >= 1, "float-controls extension present");
        assert_eq!(modes, 1, "one execution mode for the single entry point");
    }

    /// Fail-closed: garbage input is refused, never 'fixed'.
    #[test]
    fn injector_refuses_non_spirv() {
        assert!(inject_denorm_preserve(&[0xdead_beef; 8]).is_none());
        assert!(inject_denorm_preserve(&[]).is_none());
    }
}
