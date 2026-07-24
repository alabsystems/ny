// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Robust wgpu error handling so GPU validation/internal/OOM errors NEVER
//! abort the process — they are turned into `Err`, letting callers fall back
//! to the sound CPU reference path (#live-bug: acasxu wgpu validation panic).
//!
//! ## Why this exists
//!
//! By default wgpu's uncaptured-error handler *panics* (`default_error_handler`
//! in `wgpu-28.0.0/src/backend/wgpu_core.rs:694`: `panic!("wgpu error: {err}")`).
//! That panic is raised on an internal wgpu thread and cannot be caught at the
//! caller's operation boundary; even with the release profile's unwinding it
//! can tear down the GPU worker and poison later work. A confirmed trigger is a
//! small ACAS-Xu fully-connected net on the Metal backend.
//!
//! ## Two complementary mechanisms
//!
//! 1. **Error scopes** ([`WgpuDevice::run_gpu_checked`]). Before doing GPU work
//!    that may raise a validation/internal/OOM error (bind-group creation, pass
//!    encoding, `queue.submit`, readback `poll`), we push error scopes. After
//!    the work we pop them; if any captured an error we return `Err` instead of
//!    letting it reach the (panicking) uncaptured handler. Error scopes in wgpu
//!    are **thread-local**, so this only catches errors reported on the calling
//!    thread — which is where synchronous validation errors surface.
//!
//! 2. **A non-panicking uncaptured-error backstop**
//!    ([`UncapturedErrorState`]). Installed once via
//!    `device.on_uncaptured_error(...)`. Any error that escapes the thread-local
//!    scopes (e.g. surfaced asynchronously, or on a wgpu-internal thread) is
//!    logged and *recorded* in a shared flag instead of aborting. The flag is
//!    checked by [`WgpuDevice::run_gpu_checked`] so that an op whose error was
//!    delivered out-of-band still returns `Err` rather than silently returning
//!    stale/garbage buffer contents.
//!
//! ## Soundness
//!
//! A GPU error must never be presented as a valid result. On any captured
//! error we return `Err`; the ny-propagate callers
//! (`Linear::propagate_ibp_with_engine`, `propagate_linear_with_engine`,
//! sequential CROWN `crown_backward_gpu`, etc.) already fall back to the CPU
//! reference implementation on engine `Err`, which produces the correct,
//! sound verdict.

use super::WgpuDevice;
use ny_core::{NyError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tracing::error;

/// Shared state recording uncaptured wgpu errors as a non-panicking backstop.
///
/// Installed as the device's `on_uncaptured_error` handler. Increments a
/// generation counter and stores the most recent error message whenever wgpu
/// reports an error that no thread-local error scope caught. [`WgpuDevice`]
/// snapshots the counter before a GPU op and compares afterwards to detect
/// out-of-band errors.
#[derive(Debug, Default)]
pub(super) struct UncapturedErrorState {
    /// Monotonic count of uncaptured errors seen so far.
    generation: AtomicU64,
    /// Most recent uncaptured error message (for diagnostics).
    last_message: Mutex<Option<String>>,
}

impl UncapturedErrorState {
    /// Record an uncaptured error: log it and bump the generation counter.
    ///
    /// MUST NOT panic — this runs from wgpu's error path, possibly on an
    /// internal wgpu thread, where a second panic could tear down the worker
    /// and poison subsequent GPU work.
    fn record(&self, message: String) {
        error!(target: "ny_gpu::wgpu", "uncaptured wgpu error (handled as recoverable, will fall back to CPU): {message}");
        if let Ok(mut guard) = self.last_message.lock() {
            *guard = Some(message);
        }
        // Bump after storing the message so a reader that observes the new
        // generation also sees the message (best-effort diagnostics).
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Snapshot the current generation counter.
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Take the most recent recorded error message, if any.
    fn last_message(&self) -> Option<String> {
        self.last_message
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }
}

impl WgpuDevice {
    /// Install the non-panicking uncaptured-error backstop on a freshly created
    /// device and return the shared state for storage in [`WgpuDevice`].
    ///
    /// Replaces wgpu's default handler (which panics/aborts the process) with a
    /// logger that records the error so the in-flight op can be failed cleanly.
    pub(super) fn install_uncaptured_error_handler(
        device: &wgpu::Device,
    ) -> Arc<UncapturedErrorState> {
        let state = Arc::new(UncapturedErrorState::default());
        let handler_state = Arc::clone(&state);
        device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
            handler_state.record(format!("{err}"));
        }));
        state
    }

    /// Run a GPU operation, capturing any wgpu validation/internal/OOM error and
    /// returning it as `Err` instead of letting it abort the process.
    ///
    /// Pushes validation, internal, and OOM error scopes on the *calling thread*
    /// (the same thread that submits work and polls for readback in our
    /// synchronous GPU paths), runs `op`, then pops the scopes and inspects them.
    /// Also compares the uncaptured-error generation counter before/after to
    /// catch errors delivered to the backstop handler out-of-band.
    ///
    /// On any GPU error, returns `Err` so the caller can fall back to CPU. The
    /// returned value from a successful `op` is forwarded unchanged.
    ///
    /// Soundness: a GPU error never yields a valid-looking value — we either
    /// return the `op` error or a fresh error describing the GPU failure.
    pub(super) fn run_gpu_checked<T>(
        &self,
        label: &str,
        op: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        // Serialize GPU submission+readback so concurrent Rayon BaB workers do
        // not race on shared device state (pooled/cached buffers, error scopes).
        // A poisoned lock means a prior GPU op panicked mid-flight while holding
        // it — recover the guard and continue (the panic, if any, was already
        // contained; failing to CPU is the safe choice but recovering keeps the
        // GPU path usable). See the `gpu_serialize` field doc.
        let _guard = self
            .gpu_serialize
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let start_generation = self.uncaptured_errors.generation();

        let result = run_gpu_checked_on_device(&self.device, label, op);

        // Backstop: an error delivered to the uncaptured handler (e.g. on a
        // wgpu-internal thread, or asynchronously) won't appear in the scopes.
        // If the generation advanced, fail the op so we fall back to CPU.
        if result.is_ok() && self.uncaptured_errors.generation() != start_generation {
            let detail = self
                .uncaptured_errors
                .last_message()
                .unwrap_or_else(|| "unknown".to_string());
            return Err(NyError::InternalError(format!(
                "uncaptured wgpu error during {label} (recoverable; falling back to CPU): {detail}"
            )));
        }

        result
    }
}

/// Run `op` inside wgpu validation/internal/OOM error scopes on the calling
/// thread, turning any captured GPU error into `Err` instead of a process abort.
///
/// This is the device-level core used by [`WgpuDevice::run_gpu_checked`] and by
/// the cached IBP-forward plan structs (which hold their own `wgpu::Device`
/// clone but not the [`WgpuDevice`] wrapper). The non-panicking
/// `on_uncaptured_error` backstop is installed on the device itself, so it
/// protects callers reached through any clone of the same device.
///
/// Soundness: on any captured error we return `Err`; the result of a successful
/// `op` is forwarded unchanged so a GPU failure never masquerades as a value.
pub(super) fn run_gpu_checked_on_device<T>(
    device: &wgpu::Device,
    label: &str,
    op: impl FnOnce() -> Result<T>,
) -> Result<T> {
    // Push in OOM, Internal, Validation order so the innermost (Validation)
    // is popped first; each filter routes its matching error kind.
    let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let op_result = op();

    // Always pop all scopes (and await) regardless of op outcome, so the
    // thread-local scope stack stays balanced for subsequent ops.
    let validation_err = pollster::block_on(validation_scope.pop());
    let internal_err = pollster::block_on(internal_scope.pop());
    let oom_err = pollster::block_on(oom_scope.pop());

    if let Some(err) = validation_err {
        return Err(NyError::InternalError(format!(
            "wgpu validation error during {label} (recoverable; falling back to CPU): {err}"
        )));
    }
    if let Some(err) = internal_err {
        return Err(NyError::InternalError(format!(
            "wgpu internal error during {label} (recoverable; falling back to CPU): {err}"
        )));
    }
    if let Some(err) = oom_err {
        return Err(NyError::InternalError(format!(
            "wgpu out-of-memory during {label} (recoverable; falling back to CPU): {err}"
        )));
    }

    op_result
}
