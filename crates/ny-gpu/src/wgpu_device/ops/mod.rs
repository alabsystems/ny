// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! WgpuDevice operation implementations split by GPU compute concern.

mod attention;
mod conv_transpose;
pub(super) mod conv_transpose_plan;
mod crown_backward;
mod crown_backward_encode;
mod crown_backward_sound_host;
mod crown_backward_sound_resident;
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
mod eft_selfcheck;
mod f32_selfcheck;
mod gemm;
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
mod maxpool_crown_sound;
pub(super) mod point_vjp_resident;
pub(super) mod resident_weights;
mod softmax;
mod traits;

#[cfg(test)]
mod crown_backward_tests;
// Standalone MAKE-OR-BREAK experiment (wired into NO verdict path): does the live
// Metal/wgpu compiler preserve the error-free transforms double-single depends on?
// Only compiles under the gpu-tests build, so a plain build/verdict run never sees it.
#[cfg(all(test, feature = "gpu-tests"))]
mod double_single_probe;
#[cfg(test)]
mod tests;

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
