// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backend and model routing helpers for `ny beta-crown`.

use super::super::backend::{apply_preset_device, BackendRequest, BackendRequestSource};
use crate::{BackendArg, CompleteVerifierArg};

/// Input element-count threshold (in scalars) above which the GPU is preferred by
/// the auto-backend default.
///
/// Measured on real Metal hardware: the wgpu backend accelerates the large conv
/// GEMMs in image/large-net models (cifar100=3072, traffic_signs=12288,
/// tinyimagenet=9408, yolo=8112, vit=3072, vggnet16=150528 input elements) but
/// HURTS small / low-dimensional models, where the serialized GPU op lock costs
/// more than the highly-parallel CPU input-split BaB saves (acasxu=5, cersyve=4,
/// sat_relu=30, collins_rul=400 input elements → CPU). A strict `> 1000` split
/// cleanly separates these two regimes.
pub(crate) const AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS: usize = 1000;

/// Pure auto-backend default decision used when neither an explicit `--backend`,
/// the legacy `--gpu` flag, nor a preset `general.device` selected a backend.
///
/// Chooses wgpu (GPU) when the model input is LARGE (`input_element_count`
/// strictly greater than [`AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS`]) AND a GPU
/// backend is available; otherwise CPU.
///
/// SOUNDNESS: the backend is numerically faithful (CPU and GPU produce the same
/// verified/unsat/sat verdict), so this is purely a speed choice and can never
/// change a verdict. When the input size is unknown (e.g. epsilon-ball mode with
/// no VNN-LIB spec) we conservatively prefer CPU, which is always available and
/// never worse for the small/low-dim nets that lack a spec to size.
#[must_use]
pub(super) fn auto_backend_default(
    input_element_count: Option<usize>,
    gpu_available: bool,
) -> (BackendArg, &'static str) {
    match input_element_count {
        Some(count) if count > AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS && gpu_available => (
            BackendArg::Wgpu,
            "large input (>1000 elements): conv GEMMs dominate, GPU faster",
        ),
        Some(count) if count > AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS => (
            BackendArg::Cpu,
            "large input but no GPU available: using CPU",
        ),
        Some(_) => (
            BackendArg::Cpu,
            "small/low-dim input (<=1000 elements): CPU input-split BaB faster",
        ),
        None => (
            BackendArg::Cpu,
            "input size unknown pre-load: defaulting to CPU",
        ),
    }
}

/// Resolve the effective β-CROWN backend.
///
/// Precedence (highest first):
/// 1. explicit `--backend cpu|wgpu`
/// 2. legacy `--gpu` flag (→ wgpu)
/// 3. preset `general.device`
/// 4. AUTO default: GPU for large inputs when available, else CPU
///    (see [`auto_backend_default`]).
///
/// `input_element_count` is the VNN-LIB spec's flattened input element count
/// (reused from the pre-load property input summary); `None`
/// when no spec is present. `gpu_available` reports whether the wgpu backend is
/// compiled in. A wgpu choice that fails to initialize a device at runtime still
/// falls back to CPU at device-creation time, so this resolution never aborts.
#[cfg(test)]
pub(super) fn resolve_beta_crown_backend(
    backend: Option<BackendArg>,
    gpu: bool,
    preset_device: Option<&str>,
    input_element_count: Option<usize>,
    gpu_available: bool,
) -> (BackendArg, Option<&'static str>) {
    let request = resolve_beta_crown_backend_request(
        backend,
        gpu,
        preset_device,
        input_element_count,
        gpu_available,
    );
    (request.backend, request.selection_reason)
}

/// Resolve the backend together with the exact source that selected it.
///
/// The older tuple helper remains for narrow callers that need only the
/// selected backend and AUTO explanation. Runtime/provenance code must use
/// this typed form so a preset request cannot be mislabeled as an explicit CLI
/// choice and a qualification fallback cannot be mislabeled as the selection.
pub(super) fn resolve_beta_crown_backend_request(
    backend: Option<BackendArg>,
    gpu: bool,
    preset_device: Option<&str>,
    input_element_count: Option<usize>,
    gpu_available: bool,
) -> BackendRequest {
    match backend {
        // 1. explicit --backend wins outright.
        Some(explicit) => BackendRequest {
            backend: explicit,
            source: BackendRequestSource::ExplicitBackend,
            selection_reason: None,
        },
        // 2. legacy --gpu flag.
        None if gpu => BackendRequest {
            backend: BackendArg::Wgpu,
            source: BackendRequestSource::LegacyGpuFlag,
            selection_reason: None,
        },
        None => {
            // 3. preset general.device, if it selected a non-default backend.
            let preset_backend = apply_preset_device(BackendArg::Cpu, false, preset_device);
            if preset_backend != BackendArg::Cpu || preset_device == Some("cpu") {
                // Preset explicitly pinned a device (wgpu, or an explicit cpu).
                return BackendRequest {
                    backend: preset_backend,
                    source: BackendRequestSource::Preset,
                    selection_reason: None,
                };
            }
            // 4. AUTO default — no explicit/legacy/preset signal.
            let (chosen, reason) = auto_backend_default(input_element_count, gpu_available);
            BackendRequest {
                backend: chosen,
                source: BackendRequestSource::Auto,
                selection_reason: Some(reason),
            }
        }
    }
}

/// Whether the caller deliberately selected WGPU rather than reaching it via AUTO.
///
/// Kept as a pure regression helper: runtime attack routing now preserves the
/// complete pre-policy typed request instead of re-deriving this narrower bit.
#[cfg(test)]
#[must_use]
pub(super) fn explicitly_requests_wgpu(
    backend: Option<BackendArg>,
    gpu: bool,
    preset_device: Option<&str>,
) -> bool {
    backend == Some(BackendArg::Wgpu)
        || (backend.is_none() && gpu)
        || (backend.is_none() && !gpu && preset_device == Some("wgpu"))
}

pub(super) fn route_conv_model_to_graph(
    has_conv2d: bool,
    complete_verifier: CompleteVerifierArg,
    use_relu_split: bool,
    is_input_split: bool,
    explicit_branching: bool,
    use_alpha: bool,
    use_patches_mode: bool,
) -> (bool, bool) {
    if !has_conv2d {
        return (false, use_relu_split);
    }

    if complete_verifier == CompleteVerifierArg::Mip {
        return (true, true);
    }

    if !use_patches_mode {
        if is_input_split || use_relu_split {
            return (true, use_relu_split);
        }
        // #3813: sequential Conv2d β-CROWN ignores conv_mode, so matrix-mode
        // presets must use the graph ReLU-split engine to honor the reference
        // cuts => matrix policy.
        return (true, true);
    }

    if use_relu_split || is_input_split {
        return (true, use_relu_split);
    }

    if !explicit_branching && use_alpha {
        return (true, true);
    }

    (false, use_relu_split)
}
