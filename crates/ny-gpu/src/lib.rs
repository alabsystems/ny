// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Accelerated bound propagation for ny.
//!
//! This crate provides optimized implementations of the core bound propagation
//! operations, using SIMD, parallel execution, and GPU compute for performance.
//!
//! ## Available Backends
//!
//! - **CPU (Rayon)**: Parallel CPU implementation with auto-vectorization
//! - **wgpu**: Cross-platform GPU compute via WebGPU (Metal, Vulkan, DX12)
//!
//! ## Public API Boundary
//!
//! This crate exports acceleration backends and device interfaces only.
//! Algorithm-internal domain types remain owned by `ny-propagate` and must
//! not be re-exported from `ny-gpu`.
//!
//! ## Design
//!
//! The main acceleration targets are:
//! 1. **Linear layer IBP** - Matrix-vector operations with interval arithmetic
//! 2. **MatMul IBP** - Batched matrix multiplication with interval bounds
//! 3. **Per-position CROWN** - Independent CROWN execution per sequence position
//! 4. **Full CROWN backward** - GPU-resident backward pass (wgpu)
//! 5. **GPU IBP forward** - GPU-resident forward IBP plan (wgpu)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ny_gpu::WgpuDevice;
//!
//! // GPU acceleration (wgpu - cross-platform)
//! let gpu_device = WgpuDevice::new().unwrap();
//! // Use gpu_device for accelerated bound propagation
//! ```

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

/// GPU buffer management for batched linear constraints in Clip-and-Verify.
#[cfg(feature = "wgpu")]
pub mod constraint_buffers;
/// WGSL shader code fragments for accessing packed constraint data from GPU buffers.
#[cfg(feature = "wgpu")]
pub mod constraint_shaders;
/// GPU-accelerated bound propagation using wgpu (WebGPU) compute shaders.
#[cfg(feature = "wgpu")]
pub mod wgpu_device;

/// GPU-side constraint buffer manager for batched Clip-and-Verify.
#[cfg(feature = "wgpu")]
pub use constraint_buffers::GpuConstraintBuffers;
/// WebGPU device for cross-platform GPU-accelerated bound propagation.
#[cfg(feature = "wgpu")]
pub use wgpu_device::WgpuDevice;

mod accelerated;
mod backend;

#[cfg(feature = "wgpu")]
#[doc(hidden)]
pub mod benchmark_support;

/// Re-export the canonical FALLBACK_BOUND from ny-core for crate-internal use.
/// GPU WGSL shaders embed this as a literal `1e10`; the contract test
/// `test_fallback_bound_consistent` verifies the values match.
pub(crate) use ny_core::FALLBACK_BOUND;

#[cfg(test)]
mod tests;

// Bound-correctness tests for the C-matrix-seeded GPU resnet ROOT pass
// (#w4-root-gpu). Real-device tests: compiled only under `gpu-tests`.
#[cfg(all(test, feature = "gpu-tests"))]
mod spec_root_gpu_tests;

// Parity + never-looser tests for the GPU per-domain β optimization
// (#w4-split-tightening). Real-device tests: compiled only under `gpu-tests`.
#[cfg(all(test, feature = "gpu-tests"))]
mod beta_grad_gpu_tests;

// Enclosure oracle for per-subdomain intermediate-bound refinement
// (#interm-refine). Real-device tests: compiled only under `gpu-tests`.
#[cfg(all(test, feature = "gpu-tests"))]
mod interm_refine_gpu_tests;

/// Accelerated per-position CROWN verification with parallel and sequential execution modes.
pub use accelerated::{
    crown_per_position_parallel, crown_per_position_parallel_with_engine,
    crown_per_position_sequential_with_engine, AcceleratedBoundPropagation, AcceleratedDevice,
};
/// Backend trait and compute device abstraction for GPU/CPU dispatch.
pub use backend::{
    shared_cpu_engine, wgpu_adapter_available, wgpu_backend_compiled, Backend, ComputeDevice,
};
