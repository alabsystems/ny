// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backend selection helpers shared across CLI commands.

use ny_core::GemmEngine;
use ny_gpu::ComputeDevice;
use tracing::warn;

use crate::BackendArg;

/// Resolve the effective backend from --backend and --gpu flags.
///
/// --backend takes precedence. If --backend is not specified (default Cpu),
/// but --gpu is true, use wgpu for backward compatibility.
pub(crate) fn resolve_backend(backend: BackendArg, gpu: bool) -> BackendArg {
    if backend != BackendArg::Cpu {
        // --backend was explicitly specified
        backend
    } else if gpu {
        // Legacy --gpu flag, use wgpu for backward compat
        BackendArg::Wgpu
    } else {
        BackendArg::Cpu
    }
}

/// Apply preset `general.device` as a fallback when no CLI flag overrides.
///
/// Precedence: --backend > --gpu > preset general.device > CPU default.
/// Only activates when `cli_backend` is CPU and `gpu` is false (no explicit CLI override).
pub(crate) fn apply_preset_device(
    cli_backend: BackendArg,
    gpu: bool,
    preset_device: Option<&str>,
) -> BackendArg {
    if cli_backend != BackendArg::Cpu || gpu {
        // CLI explicitly set a backend — preset cannot override
        return cli_backend;
    }
    match preset_device {
        Some("wgpu") => BackendArg::Wgpu,
        Some("cpu") | None => BackendArg::Cpu,
        Some(other) => {
            warn!("Unknown preset device '{}', using CPU", other);
            BackendArg::Cpu
        }
    }
}

pub(crate) struct GemmBackendResolution<T> {
    pub(crate) backend: BackendArg,
    pub(crate) device: Option<T>,
}

impl<T> GemmBackendResolution<T> {
    pub(crate) const fn cpu() -> Self {
        Self {
            backend: BackendArg::Cpu,
            device: None,
        }
    }
}

impl<T: GemmEngine> GemmBackendResolution<T> {
    pub(crate) fn gemm_engine(&self) -> Option<&dyn GemmEngine> {
        self.device.as_ref().map(|device| device as &dyn GemmEngine)
    }
}

pub(crate) fn resolve_gemm_backend_with_factory<T, F>(
    backend: BackendArg,
    gpu: bool,
    json: bool,
    build_device: F,
) -> GemmBackendResolution<T>
where
    F: FnOnce(BackendArg) -> anyhow::Result<T>,
{
    let mut effective_backend = resolve_backend(backend, gpu);
    let device = match effective_backend {
        BackendArg::Cpu => None,
        BackendArg::Wgpu => match build_device(effective_backend) {
            Ok(device) => Some(device),
            Err(error) => {
                if !json {
                    warn!("WGPU backend not available: {error}. Using CPU.");
                }
                None
            }
        },
    };

    if device.is_none() && effective_backend != BackendArg::Cpu {
        effective_backend = BackendArg::Cpu;
    }

    GemmBackendResolution {
        backend: effective_backend,
        device,
    }
}

pub(crate) fn resolve_gemm_backend(
    backend: BackendArg,
    gpu: bool,
    json: bool,
) -> GemmBackendResolution<ComputeDevice> {
    resolve_gemm_backend_with_factory(backend, gpu, json, |effective_backend| {
        Ok(ComputeDevice::new(effective_backend.into())?)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_preset_device, resolve_backend, resolve_gemm_backend_with_factory,
        GemmBackendResolution,
    };
    use crate::BackendArg;

    #[test]
    fn test_resolve_backend_explicit_wgpu() {
        let result = resolve_backend(BackendArg::Wgpu, false);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_explicit_wgpu_with_gpu_flag() {
        // --backend takes precedence over --gpu
        let result = resolve_backend(BackendArg::Wgpu, true);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_legacy_gpu_flag() {
        // When --backend is default (Cpu) and --gpu is true, use wgpu
        let result = resolve_backend(BackendArg::Cpu, true);
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_resolve_backend_default_cpu() {
        let result = resolve_backend(BackendArg::Cpu, false);
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_resolve_gemm_backend_with_factory_keeps_cpu_without_device() {
        let resolved: GemmBackendResolution<()> =
            resolve_gemm_backend_with_factory(BackendArg::Cpu, false, false, |_| {
                panic!("cpu backend should not build a device")
            });
        assert_eq!(resolved.backend, BackendArg::Cpu);
        assert!(resolved.device.is_none());
    }

    #[test]
    fn test_resolve_gemm_backend_with_factory_builds_non_cpu_device() {
        let resolved =
            resolve_gemm_backend_with_factory(BackendArg::Wgpu, false, false, |_| Ok(()));
        assert_eq!(resolved.backend, BackendArg::Wgpu);
        assert!(resolved.device.is_some());
    }

    #[test]
    fn test_resolve_gemm_backend_with_factory_falls_back_on_error() {
        let resolved: GemmBackendResolution<()> =
            resolve_gemm_backend_with_factory(BackendArg::Wgpu, false, true, |_| {
                Err(anyhow::anyhow!("wgpu unavailable"))
            });
        assert_eq!(resolved.backend, BackendArg::Cpu);
        assert!(resolved.device.is_none());
    }

    #[test]
    fn test_apply_preset_device_wgpu_when_no_cli_override() {
        // Preset `device: wgpu` activates GPU when CLI uses defaults
        let result = apply_preset_device(BackendArg::Cpu, false, Some("wgpu"));
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_apply_preset_device_mlx_falls_back_to_cpu() {
        // mlx is no longer a supported backend — falls back to CPU
        let result = apply_preset_device(BackendArg::Cpu, false, Some("mlx"));
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_apply_preset_device_cli_backend_takes_precedence() {
        // --backend wgpu already set — preset cpu cannot downgrade
        let result = apply_preset_device(BackendArg::Wgpu, false, Some("cpu"));
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_apply_preset_device_gpu_flag_takes_precedence() {
        // --gpu flag already set (resolved to Wgpu) — preset cpu cannot downgrade
        let result = apply_preset_device(BackendArg::Wgpu, true, Some("cpu"));
        assert_eq!(result, BackendArg::Wgpu);
    }

    #[test]
    fn test_apply_preset_device_none_stays_cpu() {
        // No preset device — stays CPU
        let result = apply_preset_device(BackendArg::Cpu, false, None);
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_apply_preset_device_unknown_falls_back_cpu() {
        // Unknown device string — falls back to CPU with warning
        let result = apply_preset_device(BackendArg::Cpu, false, Some("vulkan"));
        assert_eq!(result, BackendArg::Cpu);
    }

    #[test]
    fn test_apply_preset_device_explicit_cpu() {
        // Preset says cpu — stays CPU
        let result = apply_preset_device(BackendArg::Cpu, false, Some("cpu"));
        assert_eq!(result, BackendArg::Cpu);
    }
}
