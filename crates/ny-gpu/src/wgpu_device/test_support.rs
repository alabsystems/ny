// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared GPU test helpers for wgpu-backed integration tests.

use super::{WgpuDevice, WgpuVerdictRequest};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

static GPU_DEVICE: OnceLock<Result<Arc<WgpuDevice>, String>> = OnceLock::new();
static GPU_VERDICT_DEVICE: OnceLock<Result<Arc<WgpuDevice>, String>> = OnceLock::new();
static GPU_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Get the shared GPU device. Fails the test when GPU is unavailable.
pub(crate) fn require_device() -> Arc<WgpuDevice> {
    match GPU_DEVICE.get_or_init(|| match WgpuDevice::new() {
        Ok(device) => Ok(Arc::new(device)),
        Err(error) => Err(error.to_string()),
    }) {
        Ok(device) => Arc::clone(device),
        Err(error) => {
            panic!(
                "GPU required but not available: {error}. \
                 Run with --features gpu-tests only when GPU hardware is present."
            );
        }
    }
}

/// Get the shared explicitly qualified verdict device. A rejected ladder is a
/// hard test failure with the typed report preserved in the message.
pub(crate) fn require_verdict_device() -> Arc<WgpuDevice> {
    match GPU_VERDICT_DEVICE.get_or_init(|| {
        WgpuDevice::new_for_verdict(WgpuVerdictRequest::new())
            .map(Arc::new)
            .map_err(|error| error.to_string())
    }) {
        Ok(device) => Arc::clone(device),
        Err(error) => panic!(
            "verdict-qualified GPU required but qualification failed: {error}. \
             Run with --features gpu-tests only on a conformant GPU host."
        ),
    }
}

/// Serialize GPU tests that share global device/process state.
///
/// Poisoning is IGNORED: the mutex protects no data, it only serializes device
/// access, so one failing test's panic-while-locked says nothing about the
/// mutex's (empty) state. Honoring poison here cascaded every later gpu test
/// in the process into a spurious `PoisonError` failure.
pub(crate) fn gpu_test_serial_guard() -> MutexGuard<'static, ()> {
    GPU_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether the verdict-bearing raw-WGPU CROWN lane is unavailable under the
/// current source gate, explicit typed request, and live probe ladder.
///
/// This is a fail-closed consistency check, not a test-skip helper: it asserts
/// that the `GemmEngine` accessor and `GpuCrownBackward` capability claim move
/// together. GPU conformance tests that require the route must assert the
/// complement and fail when the selected adapter cannot qualify.
pub(crate) fn sound_gpu_crown_quarantined(device: &WgpuDevice) -> bool {
    if ny_core::GemmEngine::as_gpu_crown_backward(device).is_some() {
        return false;
    }
    assert!(
        !ny_core::GpuCrownBackward::provides_sound_gpu_crown(device),
        "inconsistent quarantine: GemmEngine exposes no GPU CROWN route while \
         provides_sound_gpu_crown() still claims to offer one"
    );
    true
}
