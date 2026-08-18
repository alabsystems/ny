// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-global optional **IEEE round-to-nearest f32** GEMM accelerator
//! (e.g. CUDA cuBLAS `Sgemm`) — the f32 sibling of [`crate::sound_f64_gemm`].
//!
//! The dominant full-vnncomp compute (59–66% profiled) is f32 GEMM in the IBP
//! forward and PGD/attack paths. Those paths reach a GPU only through the
//! engine seam (`ny_core::GemmEngine::gemm_f32` on the backend `ComputeDevice`),
//! which historically dispatched to the wgpu WGSL shader or CPU only. cuBLAS
//! `Sgemm` on the same GPU is measured 2–3.4× faster than the WGSL engine at
//! every hotspot shape (and up to 29× vs the in-rayon single-threaded faer
//! path), so the backend consults this global to offload its `gemm_f32`
//! traffic to cuBLAS when available.
//!
//! # Soundness contract
//!
//! The installed engine MUST compute plain IEEE round-to-nearest f32 GEMM (no
//! TF32 / BF16x9 / fast-math; `ny_cuda::CudaGemmEngine` pins
//! `CUBLAS_DEFAULT_MATH` and refuses emulation environments). Verdict-feeding
//! engine call sites certify engine results with summation-order-independent
//! bounds (the linear IBP path widens by `in_features + 2` ULP, the conv IBP
//! path equivalently), so substituting one RN-f32 summation order for another
//! is covered. Attack/PGD call sites need no error accounting at all
//! (counterexamples are re-checked concretely).
//!
//! # Initialization policy
//!
//! Same factory pattern as [`crate::sound_f64_gemm`]: the factory is invoked
//! once, on the first offloadable GEMM, so instances whose engine traffic never
//! qualifies do not pay GPU context/handle initialization. When the CLI is
//! built with `--features cuda`, both globals share one engine instance (one
//! CUDA context + cuBLAS handle). Unbounded library calls retain that lazy
//! behavior. The VNN-COMP CLI route explicitly prewarms an enabled CUDA factory
//! before command dispatch, and finite-deadline library calls observe only an
//! already-published engine so driver initialization can never consume verifier
//! authority.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};
use std::time::Instant;

use ny_core::GemmEngine;

type SharedEngine = Arc<dyn GemmEngine>;
type Factory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;

/// Installed factory (set once at startup; cheap — no device init).
static FACTORY: OnceLock<Factory> = OnceLock::new();
/// Lazily-built engine, materialized from the factory on first use.
static ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();
/// Process-global count of actual calls through the installed engine. This is
/// provenance telemetry, not a call-local ownership claim.
static CALLS: AtomicU64 = AtomicU64::new(0);

/// Non-forcing snapshot of the process-global fast-f32 engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastF32GemmTelemetry {
    /// Calls actually issued through the materialized engine.
    pub calls: u64,
    /// Backend identity reported by that engine. `None` means the factory has
    /// not materialized a usable engine; it must not be described as CUDA.
    pub backend: Option<&'static str>,
}

/// Install a process-global factory for an IEEE RN-f32 GEMM accelerator. The
/// factory is invoked at most once (on the first offloadable `gemm_f32`); it
/// should construct an engine whose `gemm_f32` is plain IEEE round-to-nearest
/// f32 (e.g. cuBLAS `Sgemm` in `CUBLAS_DEFAULT_MATH`), or return `None` if
/// unavailable. First installation wins (idempotent).
pub fn set_fast_f32_gemm_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// Directly install a concrete engine (wraps it in a trivial factory). Useful
/// for tests / non-lazy callers.
pub fn set_fast_f32_gemm_engine(engine: SharedEngine) {
    set_fast_f32_gemm_factory(move || Some(engine.clone()));
}

/// Whether a factory is installed (does NOT force engine construction).
#[must_use]
pub fn is_installed() -> bool {
    FACTORY.get().is_some()
}

fn resolve_engine_for_deadline<'a>(
    deadline: Option<Instant>,
    factory: &'a OnceLock<Factory>,
    engine: &'a OnceLock<Option<SharedEngine>>,
) -> Option<&'a SharedEngine> {
    if deadline.is_some() {
        // `OnceLock::get` neither initializes nor waits for an in-progress
        // initializer. Finite optional work therefore fails closed immediately
        // when the accelerator has not already been published.
        engine.get().and_then(Option::as_ref)
    } else {
        engine
            .get_or_init(|| factory.get().and_then(|factory| factory()))
            .as_ref()
    }
}

/// Whether a usable engine has already been published.
///
/// Unlike [`is_installed`], this never treats a merely registered cold factory
/// as available and never initializes it.
#[must_use]
pub(crate) fn is_preinitialized() -> bool {
    ENGINE.get().is_some_and(Option::is_some)
}

/// Materialize and clone the process-global fast-f32 engine.
///
/// This is the owned-handle sibling of [`with_engine`]. It exists for
/// long-lived, attack-only steering channels that must borrow an engine across
/// many call frames. The CLI's CUDA build installs the same shared engine into
/// the f32/f64 factories, so this avoids constructing a second graphics API
/// device merely to accelerate falsification. A failed factory stays cached as
/// `None`; callers must fall back without treating failure as a verdict.
#[must_use]
pub fn shared_engine() -> Option<SharedEngine> {
    ENGINE
        .get_or_init(|| FACTORY.get().and_then(|factory| factory()))
        .clone()
}

/// Run `f` with the engine, lazily materializing it from the factory on first
/// call. Returns `None` when no factory is installed or it yields no engine
/// (callers then use their wgpu/CPU path).
pub fn with_engine<R>(f: impl FnOnce(&dyn GemmEngine) -> R) -> Option<R> {
    resolve_engine_for_deadline(None, &FACTORY, &ENGINE).map(|engine| {
        CALLS.fetch_add(1, Ordering::Relaxed);
        f(engine.as_ref())
    })
}

/// Run `f` only with an engine that was materialized before finite authority.
///
/// With `Some(deadline)`, this performs a nonblocking `OnceLock::get` and never
/// invokes the factory or waits for another initializer. With `None`, it
/// delegates to the historical lazy behavior of [`with_engine`].
pub(crate) fn with_engine_for_deadline<R>(
    deadline: Option<Instant>,
    f: impl FnOnce(&dyn GemmEngine) -> R,
) -> Option<R> {
    resolve_engine_for_deadline(deadline, &FACTORY, &ENGINE).map(|engine| {
        CALLS.fetch_add(1, Ordering::Relaxed);
        f(engine.as_ref())
    })
}

/// Explicitly materialize the registered fast-f32 engine before command
/// dispatch can create verifier deadline authority.
#[must_use]
pub fn prewarm_fast_f32_gemm() -> bool {
    resolve_engine_for_deadline(None, &FACTORY, &ENGINE).is_some()
}

/// Observe actual process-global usage without installing or materializing an
/// engine. A backend name is published only after an engine really exists.
#[must_use]
pub fn telemetry_snapshot() -> FastF32GemmTelemetry {
    FastF32GemmTelemetry {
        calls: CALLS.load(Ordering::Relaxed),
        backend: ENGINE
            .get()
            .and_then(Option::as_ref)
            .map(|engine| engine.backend_provenance()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn finite_resolution_never_invokes_a_cold_blocking_factory() {
        let factory: OnceLock<Factory> = OnceLock::new();
        let engine: OnceLock<Option<SharedEngine>> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                std::thread::park();
                None
            }))
            .is_ok());

        assert!(resolve_engine_for_deadline(Some(Instant::now()), &factory, &engine).is_none());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "finite resolution must not enter a cold accelerator factory"
        );
    }

    #[test]
    fn explicit_prewarm_makes_engine_ready_for_finite_resolution() {
        let factory: OnceLock<Factory> = OnceLock::new();
        let engine: OnceLock<Option<SharedEngine>> = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        assert!(factory
            .set(Box::new(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                Some(Arc::new(ny_core::NaiveCpuGemmEngine))
            }))
            .is_ok());

        let prewarmed =
            resolve_engine_for_deadline(None, &factory, &engine).expect("prewarm should publish");
        let finite = resolve_engine_for_deadline(Some(Instant::now()), &factory, &engine)
            .expect("finite lookup should observe the prewarmed engine");
        assert!(std::ptr::eq::<dyn GemmEngine>(
            prewarmed.as_ref(),
            finite.as_ref()
        ));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "finite lookup must reuse the prewarmed slot without reinvoking its factory"
        );
    }
}
