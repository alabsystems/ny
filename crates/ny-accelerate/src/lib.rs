// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_code)]

//! Apple Accelerate (vecLib BLAS) GEMM engine for ny.
//!
//! # Why
//!
//! NY's verdict-deciding CROWN backward computes `A·W` and `|A|·|W|` in f64 and
//! certifies the rounding with `γ_n·S`. On Apple silicon those folds currently
//! run through faer at a measured ~84 GFLOP/s effective, while Accelerate's
//! `cblas_dgemm` — the AMX/SME path — measures 353-454 GFLOP/s on the same box
//! for the same shapes. Faster folds do not only shorten the wall clock: they
//! convert per-node-budget IBP reversions into KEPT CROWN intermediates, i.e.
//! they buy TIGHTER BOUNDS.
//!
//! # Soundness — no new theory
//!
//! `cblas_dgemm` is an IEEE-f64 GEMM. The certified error term NY already
//! charges,
//!
//! ```text
//! |Ĉ_ij − Σ_l a_il·b_lj| ≤ γ_k · Σ_l |a_il|·|b_lj|,  γ_k = k·2⁻⁵³/(1 − k·2⁻⁵³)
//! ```
//!
//! is *summation-order independent* (Higham, Thm 3.1) and therefore already
//! bounds the error of any conventional blocked/multi-accumulator/fused inner
//! product — which is precisely what an opaque vendor BLAS needs from us. This
//! is the identical argument `ny-cuda` banks for `cublasDgemm`
//! (`crates/ny-cuda/src/lib.rs`) and `ny-propagate::faer_parallelism` banks for
//! faer's threaded reduction. This crate adds an implementation under existing
//! theory, not new theory.
//!
//! What it DOES add is the obligation to check that the theory's preconditions
//! hold on this particular opaque binary. That is the job of:
//!
//! * the one-shot [`probe`] (KA-1..KA-10 / SA-1..SA-6) — reduced precision,
//!   FTZ, DAZ, rounding mode, saturating overflow, Strassen block mixing,
//!   `beta=0` reading `C`, FPCR corruption; and
//! * three static per-call guards, G1 shape/stride (LP64 `int` ABI), G2
//!   underflow domain (the one regime `γ_n·S` cannot cover), G3 symbol identity.
//!
//! # Default OFF
//!
//! Nothing installs unless `NY_ACCELERATE_F64=1` (sound f64 seam) or
//! `NY_ACCELERATE_F32=1` (non-verdict f32 free-rider) is set, and
//! `NY_NO_ACCELERATE` disables both regardless. Un-armed, the process is
//! byte-identical to today: no probe runs, no factory is registered, no BLAS
//! threading policy is touched.
//!
//! # `unsafe`
//!
//! Confined to [`ffi`] — the only file in the crate that may use it (the crate
//! root is `#![deny(unsafe_code)]` and only that module lifts it). `ny-core`
//! keeps `#![forbid(unsafe_code)]`, `ny-propagate` keeps `#![deny(unsafe_code)]`.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
mod ffi;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod probe;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod engine;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use engine::{
    f32_seam_armed, f64_seam_armed, kill_switch_engaged, resolve_for_install, shared_engine,
    single_threaded_blas_available, telemetry, AccelerateGemmEngine, AccelerateTelemetry,
    InstallOutcome, BLAS_THREADS_ENV, DEFAULT_MIN_MACS, F32_GATE_ENV, F64_GATE_ENV,
    KILL_SWITCH_ENV, MIN_MACS_ENV,
};

// ---------------------------------------------------------------------------
// Inert stub for every other target. The public surface is identical so callers
// need no `cfg` of their own beyond choosing to depend on the crate; every
// entry point reports "not available" and nothing is ever installed.
// ---------------------------------------------------------------------------

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod stub {
    use std::sync::Arc;

    /// Kill switch env var (inert off macOS/aarch64).
    pub const KILL_SWITCH_ENV: &str = "NY_NO_ACCELERATE";
    /// Sound f64 seam gate (inert off macOS/aarch64).
    pub const F64_GATE_ENV: &str = "NY_ACCELERATE_F64";
    /// f32 free-rider gate (inert off macOS/aarch64).
    pub const F32_GATE_ENV: &str = "NY_ACCELERATE_F32";
    /// Offload floor env var (inert off macOS/aarch64).
    pub const MIN_MACS_ENV: &str = "NY_ACCELERATE_MIN_MACS";
    /// vecLib threading policy env var (inert off macOS/aarch64).
    pub const BLAS_THREADS_ENV: &str = "NY_ACCELERATE_BLAS_THREADS";
    /// Default offload floor.
    pub const DEFAULT_MIN_MACS: usize = 1 << 15;

    /// Unconstructible placeholder: Accelerate exists only on Apple silicon.
    #[derive(Debug)]
    pub enum AccelerateGemmEngine {}

    /// Seam counters (always zero off macOS/aarch64).
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AccelerateTelemetry {
        /// `gemm_f64` calls that reached `cblas_dgemm`.
        pub f64_calls: u64,
        /// `gemm_f32` calls that reached `cblas_sgemm`.
        pub f32_accelerate_calls: u64,
        /// `gemm_f32` calls that took the incumbent faer kernel.
        pub f32_faer_calls: u64,
        /// G1 refusals.
        pub declined_shape: u64,
        /// Below the MAC floor.
        pub declined_small: u64,
        /// G2 refusals.
        pub declined_underflow_domain: u64,
        /// Non-finite operand refusals.
        pub declined_non_finite: u64,
    }

    /// Outcome of an install attempt.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum InstallOutcome {
        /// Kill switch set.
        Disabled,
        /// Neither gate armed, or the platform has no Accelerate.
        NotArmed,
        /// Probe refused the host's BLAS.
        ProbeRefused(Vec<&'static str>),
        /// Installed.
        Installed {
            /// Sound f64 seam armed.
            f64_seam: bool,
            /// f32 free-rider armed.
            f32_seam: bool,
            /// Provenance record.
            summary: String,
        },
    }

    /// Always `false` off macOS/aarch64.
    #[must_use]
    pub fn kill_switch_engaged() -> bool {
        false
    }
    /// Always `false` off macOS/aarch64.
    #[must_use]
    pub fn f64_seam_armed() -> bool {
        false
    }
    /// Always `false` off macOS/aarch64.
    #[must_use]
    pub fn f32_seam_armed() -> bool {
        false
    }
    /// Always `false` off macOS/aarch64.
    #[must_use]
    pub fn single_threaded_blas_available() -> bool {
        false
    }
    /// Always `None` off macOS/aarch64.
    #[must_use]
    pub fn shared_engine() -> Option<Arc<AccelerateGemmEngine>> {
        None
    }
    /// Always zero off macOS/aarch64.
    #[must_use]
    pub fn telemetry() -> AccelerateTelemetry {
        AccelerateTelemetry::default()
    }
    /// Always `(None, NotArmed)` off macOS/aarch64.
    #[must_use]
    pub fn resolve_for_install() -> (Option<Arc<AccelerateGemmEngine>>, InstallOutcome) {
        (None, InstallOutcome::NotArmed)
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub use stub::*;
