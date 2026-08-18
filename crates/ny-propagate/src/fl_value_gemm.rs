// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-global optional **deadline-bounded RN-f32** GEMM accelerator for
//! the forward-linear VALUE seam (#fl-value-gpu-tier) — the deadline-capable
//! sibling of [`crate::fast_f32_gemm`], consulted ONLY by the f32 value-GEMM
//! dispatch under a finite deadline
//! (`forward_linear::image::certified_f64_gemm`, `allow_f32` branch).
//!
//! # Soundness contract
//!
//! The installed engine MUST implement
//! [`ny_core::GemmEngine::gemm_f32_with_deadline`] as plain IEEE
//! round-to-nearest f32 products with the bounded-dispatch deadline contract
//! (the wgpu `FlValueGemmDevice` is the intended producer). The call site
//! charges the values' accumulation error as `gamma_{K+4}^f32 · S` plus an
//! FTZ addend — summation-order independent, so any RN-f32 backend
//! (including Metal FTZ) is admissible. Engines are consulted for VALUES
//! only; the S-base GEMMs never route here.
//!
//! # Lazy + deadline-safe by design
//!
//! Same factory pattern as [`crate::sound_f64_gemm`], including its
//! deadline-safe admission: `WgpuDevice` construction (adapter + ~20 pipeline
//! compiles) can take hundreds of milliseconds, and a finite-deadline
//! verifier thread must not block on it. The factory therefore runs on a
//! background thread with a bounded admission wait; until it is ready the
//! caller keeps the tiled CPU f32 tier.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, NyError, Result};

type SharedEngine = Arc<dyn GemmEngine>;
type Factory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;

/// Installed factory (set once at startup; cheap — no device init).
static FACTORY: OnceLock<Factory> = OnceLock::new();
/// Lazily-built engine, materialized from the factory on first use.
static ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();
/// Some thread has entered (or spawned) the one-time initialization.
static INITIALIZATION_STARTED: AtomicBool = AtomicBool::new(false);
/// The bounded admission wait expired (or the background initializer failed).
static INITIALIZATION_ABANDONED: AtomicBool = AtomicBool::new(false);
/// Deadline-dispatch calls that consulted a materialized engine.
static CALLS: AtomicU64 = AtomicU64::new(0);
/// Calls whose engine result was validated and PUBLISHED by the seam (the
/// GPU tier genuinely served the value GEMM instead of the CPU tiers).
static HITS: AtomicU64 = AtomicU64::new(0);

/// Non-forcing telemetry snapshot of the process-global FL-value engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlValueGemmTelemetry {
    /// Deadline-dispatch calls issued through the materialized engine.
    pub calls: u64,
    /// Calls whose result the seam validated and published (GPU tier taken).
    pub hits: u64,
    /// Backend identity reported by the engine; `None` until the factory has
    /// materialized a usable engine.
    pub backend: Option<&'static str>,
}

/// Install a process-global factory for a deadline-bounded RN-f32 GEMM
/// accelerator. Invoked at most once, on a background thread, on the first
/// deadline-bearing f32 value GEMM. First installation wins (idempotent).
pub fn set_fl_value_gemm_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// Directly install a concrete engine (wraps it in a trivial factory).
/// Useful for tests / non-lazy callers.
pub fn set_fl_value_gemm_engine(engine: SharedEngine) {
    set_fl_value_gemm_factory(move || Some(engine.clone()));
}

/// Whether a factory is installed (does NOT force engine construction).
#[must_use]
pub fn is_installed() -> bool {
    FACTORY.get().is_some()
}

/// Record that a validated engine result was published by the seam.
pub(crate) fn record_gpu_tier_hit() {
    HITS.fetch_add(1, Ordering::Relaxed);
}

/// Observe process-global usage without installing or materializing an
/// engine. A backend name is published only after an engine really exists.
#[must_use]
pub fn telemetry_snapshot() -> FlValueGemmTelemetry {
    FlValueGemmTelemetry {
        calls: CALLS.load(Ordering::Relaxed),
        hits: HITS.load(Ordering::Relaxed),
        backend: ENGINE
            .get()
            .and_then(Option::as_ref)
            .map(|engine| engine.backend_provenance()),
    }
}

/// Deadline-safe access to the lazily initialized engine (mirror of
/// [`crate::sound_f64_gemm::with_engine_deadline`]):
///
/// 1. only non-blocking `ENGINE.get()` reads on the calling thread;
/// 2. the ordinary one-time factory starts on a background thread at most once;
/// 3. readiness is polled for a small bounded admission window; and
/// 4. `Ok(None)` (caller's CPU tiers) while initialization is unavailable.
///
/// A factory that hangs can strand only its background initializer, never the
/// deadline-bearing verifier thread.
pub(crate) fn with_engine_deadline<R>(
    deadline: Instant,
    f: impl FnOnce(&dyn GemmEngine) -> R,
) -> Result<Option<R>> {
    const MAX_INITIALIZATION_WAIT: Duration = Duration::from_secs(2);
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "fl-value f32 GEMM: deadline exceeded before engine admission".into(),
        ));
    }
    if let Some(engine) = ENGINE.get() {
        return Ok(engine.as_ref().map(|engine| {
            CALLS.fetch_add(1, Ordering::Relaxed);
            f(engine.as_ref())
        }));
    }
    if FACTORY.get().is_none() {
        return Ok(None);
    }
    if INITIALIZATION_ABANDONED.load(Ordering::Acquire) {
        return Ok(None);
    }

    if INITIALIZATION_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        let spawn = std::thread::Builder::new()
            .name("ny-fl-value-f32-init".into())
            .spawn(|| {
                let initialized = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = ENGINE.get_or_init(|| FACTORY.get().and_then(|factory| factory()));
                }));
                if initialized.is_err() && ENGINE.get().is_none() {
                    INITIALIZATION_ABANDONED.store(true, Ordering::Release);
                }
            });
        if spawn.is_err() {
            INITIALIZATION_ABANDONED.store(true, Ordering::Release);
            return Ok(None);
        }
    }

    let admission_end = deadline.min(Instant::now() + MAX_INITIALIZATION_WAIT);
    loop {
        if let Some(engine) = ENGINE.get() {
            if Instant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "fl-value f32 GEMM: deadline exceeded during engine admission".into(),
                ));
            }
            return Ok(engine.as_ref().map(|engine| {
                CALLS.fetch_add(1, Ordering::Relaxed);
                f(engine.as_ref())
            }));
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(NyError::DeadlineExceeded(
                "fl-value f32 GEMM: deadline exceeded during engine initialization".into(),
            ));
        }
        if now >= admission_end {
            INITIALIZATION_ABANDONED.store(true, Ordering::Release);
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
