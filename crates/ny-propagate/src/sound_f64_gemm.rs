// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-global optional **sound** f64 GEMM accelerator (e.g. CUDA cuBLAS).
//!
//! The verdict-deciding CPU CROWN backward computes `A·W` and `|A|·|W|` in f64
//! (see [`crate::layers::linear::crown_single::aw_f64_with_abssum`]) and certifies
//! the f64 rounding with `γ_n·S`. On a datacenter GPU, cuBLAS `Dgemm` computes the
//! SAME f64 products far faster (~18–34× vs the single-threaded CPU loop on
//! CROWN-shaped matrices). Because Higham's `γ_n·S` bound is **summation-order
//! independent**, the certified error stays valid for the cuBLAS result — verified
//! against an exact-rational oracle (0 violations across cancellation + large-k
//! cases). Crucially this accelerates the *sound CPU f64 path* itself, so it is
//! valid even under [`crate::sound_gpu_gate`] (it does NOT use the unsound wgpu
//! f32 CROWN), i.e. it speeds up competition verdicts.
//!
//! # Lazy by design
//!
//! The accelerator is installed as a **factory** (`set_sound_f64_gemm_factory`),
//! not an engine: the factory is invoked **once, on the first large `A·W`** that
//! would benefit. Easy / sat-by-attack / conv-dominated instances therefore never
//! pay the GPU context/handle initialization (~0.4s) — important for per-instance
//! VNN-COMP processes. Mirrors `sound_gpu_gate`'s process-global pattern.
//!
//! The engine is a `&dyn ny_core::GemmEngine` (trait in `ny-core`), so this crate
//! does not depend on the `unsafe` CUDA FFI crate (`ny-cuda`); the CLI installs a
//! factory that constructs the concrete `CudaGemmEngine`.

use std::sync::{Arc, OnceLock};

use ny_core::GemmEngine;

type SharedEngine = Arc<dyn GemmEngine>;
type Factory = Box<dyn Fn() -> Option<SharedEngine> + Send + Sync>;

/// Installed factory (set once at startup; cheap — no device init).
static FACTORY: OnceLock<Factory> = OnceLock::new();
/// Lazily-built engine, materialized from the factory on first use.
static ENGINE: OnceLock<Option<SharedEngine>> = OnceLock::new();

/// Install a process-global factory for a sound f64 GEMM accelerator. The factory
/// is invoked at most once (on the first large `A·W`); it should construct an
/// engine whose `gemm_f64` is exact IEEE-f64 (e.g. cuBLAS `Dgemm`) so the `γ_n·S`
/// certified-error bound remains valid, or return `None` if unavailable. First
/// installation wins (idempotent).
pub fn set_sound_f64_gemm_factory<F>(factory: F)
where
    F: Fn() -> Option<SharedEngine> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// Directly install a concrete engine (wraps it in a trivial factory). Useful for
/// tests / non-lazy callers.
pub fn set_sound_f64_gemm_engine(engine: SharedEngine) {
    set_sound_f64_gemm_factory(move || Some(engine.clone()));
}

/// Whether a factory is installed (does NOT force engine construction).
#[must_use]
pub fn is_installed() -> bool {
    FACTORY.get().is_some()
}

/// Run `f` with the engine, lazily materializing it from the factory on first
/// call. Returns `None` when no factory is installed or it yields no engine
/// (callers then use the CPU path).
pub(crate) fn with_engine<R>(f: impl FnOnce(&dyn GemmEngine) -> R) -> Option<R> {
    let engine = ENGINE.get_or_init(|| FACTORY.get().and_then(|factory| factory()));
    engine.as_ref().map(|e| f(e.as_ref()))
}
