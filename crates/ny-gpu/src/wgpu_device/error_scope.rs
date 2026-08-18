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
use std::cell::Cell;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tracing::error;

thread_local! {
    /// Exact-device token for the one deliberately reentrant checked GPU
    /// transaction.  A raw address is only an identity token; it is never
    /// dereferenced, and the owning [`GpuCheckedTransaction`] keeps the device
    /// borrowed while the token is installed.
    static CHECKED_TRANSACTION_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// One deadline-bounded, error-generation-checked GPU transaction.
///
/// Holding this guard across an intermediate sweep prevents ordinary CROWN
/// calls and working-set clears from changing retained device state between
/// cap admission and final result validation. The exact-device TLS token lets
/// only this transaction's two existing checked phase wrappers reuse the lock;
/// another device, thread, or nested transaction must acquire its own guard.
pub(super) struct GpuCheckedTransaction<'a> {
    owner: &'a WgpuDevice,
    _guard: MutexGuard<'a, ()>,
    oom_scope: Option<wgpu::ErrorScopeGuard>,
    internal_scope: Option<wgpu::ErrorScopeGuard>,
    validation_scope: Option<wgpu::ErrorScopeGuard>,
    start_generation: u64,
    deadline: std::time::Instant,
    active: bool,
}

impl GpuCheckedTransaction<'_> {
    /// Finish while the serialization guard is still held. No value may be
    /// published unless both the asynchronous-error generation and the live
    /// deadline remain valid at this final transaction boundary.
    pub(super) fn finish(&mut self, label: &str) -> Result<()> {
        self.finish_scopes(label)?;
        self.owner
            .finish_checked_transaction(label, self.start_generation, self.deadline)?;
        if !self.owner.checked_transaction_active() {
            return Err(NyError::InternalError(format!(
                "{label}: WGPU checked-transaction authority token was lost"
            )));
        }
        Ok(())
    }

    fn finish_scopes(&mut self, label: &str) -> Result<()> {
        // Every accepted sweep reaches a bounded readback before this point.
        // One final nonblocking maintain makes any synchronous validation and
        // completed-submission errors observable without spending deadline.
        let poll_error = self.owner.device.poll(wgpu::PollType::Poll).err();
        let poll_once = |scope: wgpu::ErrorScopeGuard| {
            let mut future = std::pin::pin!(scope.pop());
            let mut context = std::task::Context::from_waker(std::task::Waker::noop());
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(error) => Ok(error),
                std::task::Poll::Pending => Err(NyError::InternalError(format!(
                    "{label}: WGPU error scope was not ready after the bounded final readback"
                ))),
            }
        };
        // Pop all three in strict reverse order before interpreting any result,
        // so an error path cannot leave the caller's thread-local scope stack
        // unbalanced. These sweep-owned scopes are innermost even if a caller
        // used the public raw device to install an outer scope.
        let validation = self
            .validation_scope
            .take()
            .ok_or_else(|| NyError::InternalError(format!("{label}: validation scope missing")))
            .and_then(&poll_once);
        let internal = self
            .internal_scope
            .take()
            .ok_or_else(|| NyError::InternalError(format!("{label}: internal scope missing")))
            .and_then(&poll_once);
        let oom = self
            .oom_scope
            .take()
            .ok_or_else(|| NyError::InternalError(format!("{label}: OOM scope missing")))
            .and_then(&poll_once);

        if let Some(error) = poll_error {
            return Err(NyError::InternalError(format!(
                "{label}: nonblocking final WGPU poll failed: {error}"
            )));
        }
        for (kind, result) in [
            ("validation", validation),
            ("internal", internal),
            ("out-of-memory", oom),
        ] {
            if let Some(error) = result? {
                return Err(NyError::InternalError(format!(
                    "wgpu {kind} error during {label}: {error}"
                )));
            }
        }
        Ok(())
    }

    fn discard_scopes(&mut self) {
        for scope in [
            self.validation_scope.take(),
            self.internal_scope.take(),
            self.oom_scope.take(),
        ]
        .into_iter()
        .flatten()
        {
            // `pop` takes effect immediately. The result is irrelevant because
            // this transaction is already being discarded.
            drop(scope.pop());
        }
    }

    fn clear_token(&mut self) -> bool {
        if !self.active {
            return true;
        }
        let identity = self.owner.checked_transaction_identity();
        let matched = CHECKED_TRANSACTION_DEVICE.with(|slot| {
            let matched = slot.get() == Some(identity);
            slot.set(None);
            matched
        });
        self.active = false;
        matched
    }
}

impl Drop for GpuCheckedTransaction<'_> {
    fn drop(&mut self) {
        // Every `?` exit must release both the TLS authority and the mutex. A
        // mismatch is fail-closed for reentrancy because the slot is cleared.
        self.discard_scopes();
        let _ = self.clear_token();
    }
}

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
    fn checked_transaction_identity(&self) -> usize {
        std::ptr::from_ref(self).addr()
    }

    fn checked_transaction_active(&self) -> bool {
        let identity = self.checked_transaction_identity();
        CHECKED_TRANSACTION_DEVICE.with(|slot| slot.get() == Some(identity))
    }

    fn uncaptured_error_since(&self, label: &str, start_generation: u64) -> Result<()> {
        if self.uncaptured_errors.generation() == start_generation {
            return Ok(());
        }
        let detail = self
            .uncaptured_errors
            .last_message()
            .unwrap_or_else(|| "unknown".to_string());
        Err(NyError::InternalError(format!(
            "uncaptured wgpu error during {label} (recoverable; falling back to CPU): {detail}"
        )))
    }

    fn finish_checked_transaction(
        &self,
        label: &str,
        start_generation: u64,
        deadline: std::time::Instant,
    ) -> Result<()> {
        self.uncaptured_error_since(label, start_generation)?;
        if std::time::Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: deadline expired while finalizing the WGPU transaction"
            )));
        }
        Ok(())
    }

    /// Acquire one deadline-aware outer transaction for a multi-phase GPU
    /// operation. The returned guard installs the exact-device reentrancy token
    /// and must remain alive until all result association and validation have
    /// completed.
    pub(super) fn begin_gpu_checked_transaction(
        &self,
        label: &str,
        deadline: std::time::Instant,
    ) -> Result<GpuCheckedTransaction<'_>> {
        if CHECKED_TRANSACTION_DEVICE.with(|slot| slot.get().is_some()) {
            return Err(NyError::InternalError(format!(
                "{label}: nested WGPU checked transaction refused"
            )));
        }
        let guard = loop {
            match self.gpu_serialize.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    break poisoned.into_inner();
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(NyError::DeadlineExceeded(format!(
                            "{label}: deadline expired while waiting for the WGPU transaction lock"
                        )));
                    }
                    std::thread::yield_now();
                }
            }
        };
        if std::time::Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: deadline expired before the WGPU transaction began"
            )));
        }
        let identity = self.checked_transaction_identity();
        // Own innermost scopes for the complete transaction. This prevents an
        // outer scope installed through the public raw device accessor from
        // intercepting and hiding sweep validation/internal/OOM errors.
        let oom_scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        CHECKED_TRANSACTION_DEVICE.with(|slot| slot.set(Some(identity)));
        Ok(GpuCheckedTransaction {
            owner: self,
            _guard: guard,
            oom_scope: Some(oom_scope),
            internal_scope: Some(internal_scope),
            validation_scope: Some(validation_scope),
            start_generation: self.uncaptured_errors.generation(),
            deadline,
            active: true,
        })
    }

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
        if result.is_ok() {
            self.uncaptured_error_since(label, start_generation)?;
        }

        result
    }

    /// Deadline-aware sibling used by call-local cooperative GPU methods.
    ///
    /// Unlike [`Self::run_gpu_checked`], contention on the process-local GPU
    /// serialization lock is itself cancellable: the caller never waits past
    /// its deadline merely to discover that no scripted GPU work can start.
    pub(super) fn run_gpu_checked_with_deadline<T>(
        &self,
        label: &str,
        deadline: std::time::Instant,
        op: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let _guard = loop {
            match self.gpu_serialize.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    break poisoned.into_inner();
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(NyError::DeadlineExceeded(format!(
                            "{label}: deadline expired while waiting for the WGPU submission lock"
                        )));
                    }
                    std::thread::yield_now();
                }
            }
        };
        if std::time::Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: deadline expired before WGPU work began"
            )));
        }

        let start_generation = self.uncaptured_errors.generation();
        let result = run_gpu_checked_on_device(&self.device, label, op);
        if result.is_ok() {
            self.uncaptured_error_since(label, start_generation)?;
        }
        // Popping WGPU error scopes waits for asynchronous validation work and
        // can itself consume the remaining budget. Never publish a value after
        // that wait crossed the call-local deadline.
        if result.is_ok() && std::time::Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: deadline expired while finalizing WGPU work"
            )));
        }
        result
    }

    /// Use cancellable serialization whenever a call-local CROWN deadline is
    /// armed, preserving the legacy blocking helper for every other caller.
    pub(super) fn run_gpu_checked_with_crown_deadline<T>(
        &self,
        label: &str,
        op: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if self.checked_transaction_active() {
            let deadline = self.call_local_crown_deadline().ok_or_else(|| {
                NyError::InternalError(format!(
                    "{label}: checked WGPU transaction lacks its call-local deadline"
                ))
            })?;
            if std::time::Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(format!(
                    "{label}: deadline expired before reentrant WGPU work began"
                )));
            }
            let start_generation = self.uncaptured_errors.generation();
            // The outer transaction deliberately owns the error-generation
            // fence for the complete multi-phase call. Pushing nested wgpu
            // error scopes here would require `ErrorScope::pop()` futures,
            // whose executor wait has no deadline API. Run the phase under the
            // installed non-panicking uncaptured-error handler instead; each
            // phase's bounded readback polls the device, and both this check
            // and the outer final fence reject any delivered error before a
            // value can escape.
            let result = op();
            if result.is_ok() {
                self.uncaptured_error_since(label, start_generation)?;
                if std::time::Instant::now() >= deadline {
                    return Err(NyError::DeadlineExceeded(format!(
                        "{label}: deadline expired while finalizing reentrant WGPU work"
                    )));
                }
            }
            return result;
        }
        match self.call_local_crown_deadline() {
            Some(deadline) => self.run_gpu_checked_with_deadline(label, deadline, op),
            None => self.run_gpu_checked(label, op),
        }
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
