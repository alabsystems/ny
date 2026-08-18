// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! WgpuDevice operation implementations split by GPU compute concern.

mod attention;
pub(super) mod bab_bound_authority;
mod conv_transpose;
pub(super) mod conv_transpose_plan;
mod crown_backward;
/// #batched-bab wide-lane publication counter (observability only) — surfaced
/// for the CLI's dark `[wide-lane]` readout without widening module scope.
pub use crown_backward::wide_resnet_batched_taken_count;
mod crown_backward_encode;
mod crown_backward_sound_host;
mod crown_backward_sound_resident;
#[cfg(test)]
pub(in crate::wgpu_device) use crown_backward_sound_resident::StridedGatherParams;
mod crown_backward_types;
mod crown_concretize_sound;
mod crown_dispatch;
pub(super) mod crown_host_profile;
pub(super) mod crown_memory_estimate;
pub(super) mod crown_plan;
pub(super) mod crown_plan_key;
mod crown_plan_working;
pub(super) mod crown_timestamps;
pub(super) mod cut_fold_resident;
// Carrier-driven resident Cut-CROWN shadow (observation-only): the audited
// cut-apply kernel + host driver behind `provides_resident_cut_shadow`.
pub(super) mod cut_shadow_resident;
// `#u4` C2 device probes (TAINT_GUARD_AUDIT.md §4): the EFT min-combine taint
// twin must refuse the chain's only error-LOWERING op on a set word. The twin
// is wired into the AUTO/default production resident walk; this characterization
// module itself compiles only under the gpu-tests build. U5/U6 and B0 are now
// discharged and the raw-device source gate is open; this remains a diagnostic
// characterization module rather than an authority input of its own.
#[cfg(all(test, feature = "gpu-tests"))]
mod eft_min_combine_taint_probe;
mod eft_selfcheck;
mod f32_selfcheck;
mod gemm;
// Budget constants shared with the FL value tier (#fl-value-gpu-tier).
pub(crate) use gemm::MAX_BINDING_ELEMS;
mod ibp;
mod ibp_forward;
mod ibp_forward_plan;
mod ibp_forward_sound;
mod ibp_graph_forward;
mod ibp_graph_forward_plan;
mod ibp_graph_forward_plan_bind;
mod ibp_graph_forward_plan_build;
mod ibp_graph_forward_plan_sound;
mod ibp_graph_forward_plan_sound_bind;
mod ibp_ops_sound;
pub(in crate::wgpu_device) mod intermediate_sweep;
pub(in crate::wgpu_device) mod intermediate_sweep_carrier;
mod intermediate_sweep_dag;
pub(in crate::wgpu_device) mod intermediate_sweep_schedule;
mod maxpool_crown_sound;
pub(super) mod point_vjp_resident;
pub(super) mod resident_weights;
mod sentinel_taint_selfcheck;
mod softmax;
pub(super) mod sound_authority;
pub(in crate::wgpu_device) mod subnormal_selfcheck;
#[cfg(all(test, feature = "gpu-tests"))]
mod taint_chain;
#[cfg(all(test, feature = "gpu-tests"))]
mod taint_channel_probe;
mod traits;

#[cfg(test)]
mod crown_backward_tests;
// Standalone MAKE-OR-BREAK experiment (wired into NO verdict path): does the live
// Metal/wgpu compiler preserve the error-free transforms double-single depends on?
// Only compiles under the gpu-tests build, so a plain build/verdict run never sees it.
#[cfg(all(test, feature = "gpu-tests"))]
mod double_single_probe;
// `#u1` settling test for the PRODUCTION tiled `#eft-err` twin GEMM (per-element
// bit-compare of (V,R) against a CPU twin executing the identical sequence).
// Wired into NO verdict path; compiles only under the gpu-tests build.
#[cfg(test)]
mod tests;
#[cfg(all(test, feature = "gpu-tests"))]
mod twin_composition_probe;

use ny_tensor::{repair_inverted_bounds, InversionRepair};

/// Sanitize GPU readback buffers: NaN/Inf → ±FALLBACK_BOUND (defense-in-depth).
///
/// GPU shaders have their own NaN guards, but a subtle shader bug (workgroup race,
/// register spill) could bypass text-pattern–verified WGSL guards. This CPU-side
/// post-sanitization matches the pattern in the CPU acceleration kernels
/// (`accelerated/kernels.rs:94-107`).
///
/// Reference: #2785, #2642 (CPU kernel sanitization)
pub(super) fn sanitize_readback(lower: &mut [f32], upper: &mut [f32]) {
    use crate::FALLBACK_BOUND;
    let len = lower.len().min(upper.len());
    for i in 0..len {
        if !lower[i].is_finite() || !upper[i].is_finite() {
            lower[i] = -FALLBACK_BOUND;
            upper[i] = FALLBACK_BOUND;
        }
    }
    let _ = repair_inverted_bounds(
        &mut lower[..len],
        &mut upper[..len],
        InversionRepair::WidenToFallback(FALLBACK_BOUND),
    );
}

/// Convert a `usize` dimension to `u32` for GPU shader uniform structs.
///
/// GPU shaders (WGSL) use `u32` for dimension parameters. This helper
/// returns a descriptive error instead of silently truncating on overflow.
pub(super) fn gpu_checked_u32(value: usize, field: &str) -> ny_core::Result<u32> {
    u32::try_from(value).map_err(|_| {
        ny_core::NyError::InternalError(format!(
            "GPU shader dimension {field} = {value} exceeds u32::MAX"
        ))
    })
}
