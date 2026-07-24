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
//! # Lazy by design
//!
//! Same factory pattern as [`crate::sound_f64_gemm`]: the factory is invoked
//! once, on the first offloadable GEMM, so instances whose engine traffic never
//! qualifies do not pay GPU context/handle initialization. When the CLI is
//! built with `--features cuda`, both globals share one engine instance (one
//! CUDA context + cuBLAS handle).

use std::sync::{Arc, OnceLock};

use ny_core::GemmEngine;

type SharedEngine = Arc<dyn GemmEngine>;
type Factory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;

/// Installed factory (set once at startup; cheap — no device init).
static FACTORY: OnceLock<Factory> = OnceLock::new();
/// Lazily-built engine, materialized from the factory on first use.
static ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();

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

/// Run `f` with the engine, lazily materializing it from the factory on first
/// call. Returns `None` when no factory is installed or it yields no engine
/// (callers then use their wgpu/CPU path).
pub fn with_engine<R>(f: impl FnOnce(&dyn GemmEngine) -> R) -> Option<R> {
    let engine = ENGINE.get_or_init(|| FACTORY.get().and_then(|factory| factory()));
    engine.as_ref().map(|e| f(e.as_ref()))
}
