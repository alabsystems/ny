// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Narrow unit tests for compare/bench GemmEngine backend routing.

use super::backend::GemmBackendResolution;
use super::bench::resolve_crown_benchmark_backend_with_factory;
use super::inspect::resolve_compare_backend_with_factory;
use super::verify::resolve_effective_backend_with_factory;
use crate::BackendArg;
use ny_propagate::PropagationMethod;

#[test]
fn compare_cpu_backend_preserves_cpu_behavior() {
    let resolved: GemmBackendResolution<()> = resolve_compare_backend_with_factory(
        BackendArg::Cpu,
        false,
        false,
        PropagationMethod::Crown,
        |_| panic!("cpu compare should not build a device"),
    );
    assert_eq!(resolved.backend, BackendArg::Cpu);
    assert!(resolved.device.is_none());
}

#[test]
fn compare_crown_backend_builds_device_for_gemm_path() {
    let resolved = resolve_compare_backend_with_factory(
        BackendArg::Wgpu,
        false,
        false,
        PropagationMethod::Crown,
        |_| Ok(()),
    );
    assert_eq!(resolved.backend, BackendArg::Wgpu);
    assert!(resolved.device.is_some());
}

#[test]
fn compare_ibp_ignores_requested_gpu_backend() {
    let resolved: GemmBackendResolution<()> = resolve_compare_backend_with_factory(
        BackendArg::Wgpu,
        false,
        false,
        PropagationMethod::Ibp,
        |_| panic!("ibp compare should not build a device"),
    );
    assert_eq!(resolved.backend, BackendArg::Cpu);
    assert!(resolved.device.is_none());
}

#[test]
fn bench_cpu_backend_preserves_cpu_behavior() {
    let resolved: GemmBackendResolution<()> =
        resolve_crown_benchmark_backend_with_factory(BackendArg::Cpu, false, false, |_| {
            panic!("cpu bench should not build a device")
        });
    assert_eq!(resolved.backend, BackendArg::Cpu);
    assert!(resolved.device.is_none());
}

#[test]
fn bench_crown_backend_builds_device_for_gpu_request() {
    let resolved =
        resolve_crown_benchmark_backend_with_factory(BackendArg::Wgpu, false, false, |_| Ok(()));
    assert_eq!(resolved.backend, BackendArg::Wgpu);
    assert!(resolved.device.is_some());
}

#[test]
fn verify_threads_a_successfully_constructed_proof_device() {
    let resolved =
        resolve_effective_backend_with_factory(BackendArg::Wgpu, false, true, true, |_| Ok(()));
    assert_eq!(resolved.backend, BackendArg::Wgpu);
    assert!(resolved.use_gpu);
    assert!(resolved.device.is_some());
    assert_eq!(resolved.receipt.requested, BackendArg::Wgpu);
    assert_eq!(resolved.receipt.effective, BackendArg::Wgpu);
    assert_eq!(resolved.receipt.qualification.as_str(), "qualified");
}

#[test]
fn verify_wgpu_construction_failure_falls_back_to_cpu() {
    let resolved = resolve_effective_backend_with_factory(
        BackendArg::Wgpu,
        false,
        true,
        true,
        |_| -> anyhow::Result<u8> { Err(anyhow::anyhow!("qualification refused")) },
    );
    assert_eq!(resolved.backend, BackendArg::Cpu);
    assert!(!resolved.use_gpu);
    assert!(resolved.device.is_none());
    assert_eq!(resolved.receipt.requested, BackendArg::Wgpu);
    assert_eq!(resolved.receipt.effective, BackendArg::Cpu);
    assert_eq!(resolved.receipt.qualification.as_str(), "refused");
    assert!(resolved
        .receipt
        .fallback_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("qualification refused")));
}

#[test]
fn verify_non_crown_mode_refuses_wgpu_before_device_construction() {
    let resolved = resolve_effective_backend_with_factory(
        BackendArg::Wgpu,
        false,
        true,
        false,
        |_| -> anyhow::Result<()> {
            panic!("an IBP/layer/block/f64 mode must not construct a WGPU proof device")
        },
    );
    assert_eq!(resolved.backend, BackendArg::Cpu);
    assert!(!resolved.use_gpu);
    assert!(resolved.device.is_none());
    assert_eq!(
        resolved.receipt.failed_rung.as_deref(),
        Some("verify_mode_capability")
    );
}
