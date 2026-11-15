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
//!   as the safe path), post-processed only by inserting a capability, an
//!   extension and one execution mode per entry point, following SPIR-V layout;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PreambleSection {
    Capability,
    Extension,
    EntryPoint,
    ExecutionMode,
}

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
/// on malformed instruction extents, inspected operands, or preamble ordering.
/// This is a structural injector, not a general SPIR-V validator; production
/// input is naga's own compiler output.
pub fn inject_denorm_preserve(spirv: &[u32]) -> Option<Vec<u32>> {
    if spirv.len() < 6 || spirv[0] != SPIRV_MAGIC {
        return None;
    }
    let mut i = 5usize;
    let mut last_cap_end = None;
    let mut entry_point_ids: Vec<u32> = Vec::new();
    let mut mode_insert_at = None;
    let mut section = None;
    while i < spirv.len() {
        let instr = spirv[i];
        let opcode = instr & 0xFFFF;
        let wc = (instr >> 16) as usize;
        if wc == 0 {
            return None;
        }
        let end = i.checked_add(wc)?;
        let instruction = spirv.get(i..end)?;
        // These are the preamble classes we inspect or insert. Other
        // instructions remain unchanged; naga owns full module semantics.
        let next_section = match opcode {
            OP_CAPABILITY if wc == 2 => Some(PreambleSection::Capability),
            OP_EXTENSION if wc >= 2 => Some(PreambleSection::Extension),
            OP_ENTRY_POINT if wc >= 4 => Some(PreambleSection::EntryPoint),
            OP_EXECUTION_MODE if wc >= 3 => Some(PreambleSection::ExecutionMode),
            OP_CAPABILITY | OP_EXTENSION | OP_ENTRY_POINT | OP_EXECUTION_MODE => return None,
            _ => None,
        };
        if let Some(next) = next_section {
            if section.is_some_and(|previous| next < previous) {
                return None;
            }
            section = Some(next);
        }
        if opcode == OP_CAPABILITY {
            last_cap_end = Some(end);
        }
        if opcode == OP_ENTRY_POINT {
            entry_point_ids.push(instruction[2]);
        }
        if opcode == OP_ENTRY_POINT || opcode == OP_EXECUTION_MODE {
            mode_insert_at = Some(end);
        }
        i = end;
    }
    let cap_end = last_cap_end?;
    let mode_at = mode_insert_at?;
    if entry_point_ids.is_empty() {
        return None;
    }
    let capabilities = spirv.get(..cap_end)?;
    let preamble = spirv.get(cap_end..mode_at)?;
    let remainder = spirv.get(mode_at..)?;
    let mut out: Vec<u32> = Vec::with_capacity(spirv.len() + 16);
    out.extend_from_slice(capabilities);
    out.push(word(OP_CAPABILITY, 2));
    out.push(CAP_DENORM_PRESERVE);
    let ext = string_words("SPV_KHR_float_controls");
    out.push(word(OP_EXTENSION, 1 + ext.len() as u32));
    out.extend_from_slice(&ext);
    out.extend_from_slice(preamble);
    for entry in &entry_point_ids {
        out.push(word(OP_EXECUTION_MODE, 4));
        out.push(*entry);
        out.push(MODE_DENORM_PRESERVE);
        out.push(32);
    }
    out.extend_from_slice(remainder);
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

/// Can a device with these features on this backend load the module
/// [`create_denorm_preserving_module`] builds?
///
/// The module is SPIR-V, and `Features::PASSTHROUGH_SHADERS` alone does not
/// say a backend consumes SPIR-V. wgpu 29 selects the passthrough payload by
/// backend (`wgpu_core::device::Device::create_shader_module_passthrough`):
/// Vulkan reads `ShaderModuleDescriptorPassthrough::spirv`; Metal reads only
/// `metallib`/`msl`, DX12 only `dxil`/`hlsl`, GL only `glsl`, and each rejects
/// a SPIR-V-only descriptor with `CreateShaderModuleError::NotCompiledForBackend`
/// ("Generic shader passthrough does not contain any code compatible with this
/// backend"). Metal advertises `PASSTHROUGH_SHADERS` for its MSL route, so a
/// feature-only test sent every Apple device down a passthrough path that can
/// never load: each module fell back to plain WGSL and permanently poisoned the
/// process's DenormPreserve loading-path contract, so verdict qualification
/// refused before its ladder ran (report adapter `None`, no named rung).
///
/// This is a statement about the module FORMAT the backend accepts, not an
/// authority decision: a `true` here still proves nothing about subnormals,
/// and the verdict ladder measures the device either way.
#[must_use]
pub fn spirv_passthrough_supported(features: wgpu::Features, backend: wgpu::Backend) -> bool {
    features.contains(wgpu::Features::PASSTHROUGH_SHADERS) && backend == wgpu::Backend::Vulkan
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
    let features = device.features();
    if !features.contains(wgpu::Features::PASSTHROUGH_SHADERS) {
        return Err("PASSTHROUGH_SHADERS not enabled on this device".into());
    }
    let backend = device.adapter_info().backend;
    if !spirv_passthrough_supported(features, backend) {
        return Err(format!(
            "SPIR-V passthrough unavailable on the {backend:?} backend (wgpu consumes \
             ShaderModuleDescriptorPassthrough::spirv only on Vulkan)"
        ));
    }
    let spirv = wgsl_to_spirv(wgsl)?;
    let injected = inject_denorm_preserve(&spirv)
        .ok_or_else(|| "injection refused (module shape)".to_string())?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // SAFETY: `injected` is naga's own output plus a capability, an extension
    // and one execution mode per entry point, in SPIR-V logical order. The
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

    fn module(instructions: &[&[u32]]) -> Vec<u32> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0000, 0, 8, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
    }

    #[test]
    fn injector_refuses_short_entry_point_instructions() {
        let capability = [word(OP_CAPABILITY, 2), 1];
        let operands = [5, 1, 0];
        for count in 1..4 {
            let mut entry = vec![word(OP_ENTRY_POINT, count)];
            entry.extend_from_slice(&operands[..count as usize - 1]);
            let malformed = module(&[&capability, &entry]);
            assert!(
                inject_denorm_preserve(&malformed).is_none(),
                "entry point with only {count} words must be refused"
            );
        }
    }

    #[test]
    fn injector_refuses_malformed_preamble_instruction_sizes() {
        let capability = [word(OP_CAPABILITY, 2), 1];
        let entry = [word(OP_ENTRY_POINT, 4), 5, 1, 0];
        let malformed = [
            module(&[&[word(OP_CAPABILITY, 1)], &entry]),
            module(&[&[word(OP_CAPABILITY, 3), 1, 0], &entry]),
            module(&[&capability, &[word(OP_EXTENSION, 1)], &entry]),
            module(&[&capability, &entry, &[word(OP_EXECUTION_MODE, 1)]]),
            module(&[&capability, &entry, &[word(OP_EXECUTION_MODE, 2), 1]]),
        ];
        for (case, words) in malformed.into_iter().enumerate() {
            assert!(
                inject_denorm_preserve(&words).is_none(),
                "malformed preamble case {case} must be refused"
            );
        }
    }

    #[test]
    fn injector_refuses_capability_after_the_entry_point_preamble() {
        let mut malformed = wgsl_to_spirv(TINY).expect("compile a valid control");
        malformed.extend_from_slice(&[word(OP_CAPABILITY, 2), 1]);
        assert!(inject_denorm_preserve(&malformed).is_none());
    }

    #[test]
    fn injector_preserves_multiple_entry_points() {
        let wgsl = format!(
            "{TINY}\n@compute @workgroup_size(1)\n\
             fn second(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             data[gid.x] = data[gid.x] + 1.0;\n}}"
        );
        let spirv = wgsl_to_spirv(&wgsl).expect("compile both entry points");
        let mut entry_points = Vec::new();
        let mut i = 5;
        while i < spirv.len() {
            if spirv[i] & 0xFFFF == OP_ENTRY_POINT {
                entry_points.push(spirv[i + 2]);
            }
            i += (spirv[i] >> 16) as usize;
        }
        assert_eq!(entry_points.len(), 2);
        assert_ne!(entry_points[0], entry_points[1]);
        let mut expected: Vec<_> = entry_points.into_iter().map(|id| (id, 32)).collect();
        expected.sort_unstable();
        let injected = inject_denorm_preserve(&spirv).expect("valid preamble remains supported");
        let mut modes = Vec::new();
        let mut i = 5;
        while i < injected.len() {
            let count = (injected[i] >> 16) as usize;
            assert!(count > 0 && i + count <= injected.len());
            if injected[i] & 0xFFFF == OP_EXECUTION_MODE && injected[i + 2] == MODE_DENORM_PRESERVE
            {
                assert_eq!(count, 4);
                modes.push((injected[i + 1], injected[i + 3]));
            }
            i += count;
        }
        modes.sort_unstable();
        assert_eq!(
            modes, expected,
            "one 32-bit mode for each distinct entry point"
        );
    }

    /// CPU-only pin of the loadability predicate: the SPIR-V module loads only
    /// where wgpu reads the `spirv` payload (Vulkan with the feature granted).
    /// Metal is the case that regressed: it advertises `PASSTHROUGH_SHADERS`
    /// for MSL, and a feature-only test sent it down a route that cannot load.
    #[test]
    fn spirv_passthrough_requires_the_feature_and_the_vulkan_backend() {
        let granted = wgpu::Features::PASSTHROUGH_SHADERS;
        let none = wgpu::Features::empty();
        assert!(spirv_passthrough_supported(granted, wgpu::Backend::Vulkan));
        assert!(!spirv_passthrough_supported(none, wgpu::Backend::Vulkan));
        for backend in [
            wgpu::Backend::Metal,
            wgpu::Backend::Dx12,
            wgpu::Backend::Gl,
            wgpu::Backend::BrowserWebGpu,
            wgpu::Backend::Noop,
        ] {
            assert!(
                !spirv_passthrough_supported(granted, backend),
                "{backend:?} does not read the SPIR-V passthrough payload"
            );
            assert!(!spirv_passthrough_supported(none, backend));
        }
    }
}
