// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Process-wide compute-engine bootstrap for embedded consumers.
//!
//! ny installs a blocked-SIMD CPU GEMM as a FLOOR under the engine-gated fast
//! paths — but only from `ny-cli`'s `main()`. Every library consumer therefore
//! ran without it, and the cost is not merely speed: where an engine gates a
//! REPRESENTATION choice rather than a kernel choice, its absence changes the
//! algorithm. The conv Patches backward's batched-GEMM seam requires
//! `engine.is_some() || fast_f32_gemm::is_installed()`, so on a host with no
//! engine that seam is unreachable by construction and composition falls back
//! to a per-position scalar `conv2d_transpose` loop. The sound-f64 lane
//! likewise computes its abs-sum with a second full f64 GEMM instead of the
//! cheap f32 seam.
//!
//! [`init_process_engines`] installs both floors for consumers that never see
//! `ny-cli`.
//!
//! # Ordering
//!
//! Installation is `OnceLock`-guarded and FIRST INSTALL WINS, so call this
//! AFTER registering any accelerator you intend to use. The CPU engines are a
//! floor, never a preemption.
//!
//! # Soundness
//!
//! Both engines satisfy the plain IEEE round-to-nearest f32 contract their
//! callers assume, and every verdict-feeding caller certifies engine results
//! with a summation-order-INDEPENDENT envelope — the same argument that
//! licenses cuBLAS on this seam. Installing them can move a bound's last ULP,
//! never its validity (ny-propagate's installers record exact-rational oracle
//! validation with zero violations).

/// Install ny's CPU compute-engine floors for this process, if nothing has
/// claimed those seams already.
///
/// Idempotent, cheap, and safe to call from several places; see the module doc
/// for the ordering contract and the soundness argument. Consumers embedding
/// ny through this facade should call it once during setup — otherwise the
/// engine-gated fast paths silently choose their scalar fallbacks.
pub fn init_process_engines() {
    ny_propagate::faer_parallelism::install_cpu_gemm_engine_if_absent();
    ny_propagate::faer_parallelism::install_cpu_sound_f64_gemm_engine_if_absent();
}

#[cfg(test)]
mod tests {
    use super::init_process_engines;

    /// Idempotent, and it actually claims the f32 seam the gated fast paths
    /// test for — the whole point is that `is_installed()` becomes true.
    #[test]
    fn init_installs_the_cpu_floor_and_is_idempotent() {
        init_process_engines();
        assert!(
            ny_propagate::fast_f32_gemm::is_installed(),
            "the CPU GEMM floor must claim the seam the conv Patches backward gates on"
        );
        init_process_engines();
        assert!(ny_propagate::fast_f32_gemm::is_installed());
    }
}
