// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared GPU test helpers for wgpu-backed integration tests.

use super::WgpuDevice;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

static GPU_DEVICE: OnceLock<Result<Arc<WgpuDevice>, String>> = OnceLock::new();
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

/// Serialize GPU tests that share global device/process state.
pub(crate) fn gpu_test_serial_guard() -> MutexGuard<'static, ()> {
    GPU_TEST_MUTEX
        .lock()
        .expect("GPU test serialization lock should not be poisoned")
}
