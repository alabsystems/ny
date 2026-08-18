// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_code)]

//! Bounded tensor types with interval arithmetic.
//!
//! This crate provides tensor types where each element has lower and upper bounds,
//! supporting the bound propagation algorithms in ny.
//!
//! # Bound Representations
//!
//! - [`BoundedTensor`]: Interval bounds [lower, upper]. Simple but loose.
//! - [`zonotope::ZonotopeTensor`]: Correlation-aware bounds. Tighter for attention.
//! - [`compressed::CompressedBounds`]: f16 storage for memory efficiency.
//! - [`generic::GenericBounds`]: Consumer-facing generic wrapper (internal `f32`).
//!
//! # Memory Pooling
//!
//! - [`pool::TensorPool`]: Thread-local memory pool for buffer reuse.
//! - [`pool::PooledBuffer`]: Auto-returning buffer handle.

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

mod bounded_tensor;
/// Directed rounding for sound floating-point interval arithmetic.
pub mod rounding;

/// Compressed bounds using f16 storage for 50% memory reduction vs f32.
pub mod compressed;
/// Generic scalar-type wrapper: adapts external types (f64, f16) to internal f32 representation.
pub mod generic;
/// Multi-norm linear bounds for Lp-bounded perturbations (DeepT-style).
pub mod multi_norm;
/// Thread-local memory pool for `Vec<f32>` tensor buffers organized into power-of-2 size classes.
pub mod pool;
/// Pooled `ArrayD<f32>` storage that returns backing buffers to [`pool::TensorPool`] on drop.
pub mod pooled_array;
/// Dynamic tensor storage with append/pop for branch-and-bound search (LIFO/FIFO modes).
pub mod tensor_storage;
/// Sliding window extraction (im2col) for CNN Patches mode termination and slope unfolding.
pub mod unfold;
/// Zonotope tensor for correlation-aware bound propagation via shared error symbols.
pub mod zonotope;

/// Double-precision bounded tensor for f64 propagation (soundnessbench, sat_relu).
pub use bounded_tensor::BoundedTensor64;
/// Shared inverted-bounds repair strategy for propagation and readback code. Part of #3307.
pub use bounded_tensor::InversionRepair;
/// Optional per-normalization-slice Euclidean-ball annotation on a [`BoundedTensor`]
/// enabling exact Cauchy–Schwarz tightening at the downstream `Linear`.
pub use bounded_tensor::L2Constraint;
/// Strategy for automatic NaN/Inf repair at BoundedTensor construction. Part of #3423.
pub use bounded_tensor::RepairStrategy;
/// Shared inverted-bounds repair helpers. Part of #3307.
pub use bounded_tensor::{repair_inverted_bounds, repair_inverted_bounds_nd};
/// Interval-bounded tensor type for representing input regions with lower/upper bounds.
pub use bounded_tensor::{
    BoundedTensor, BoundedTensorHostAllocationEndpointV1, BoundedTensorHostAllocationInvalidV1,
    BoundedTensorHostAllocationProvenanceV1, BoundedTensorHostAllocationReceiptV1,
    BoundedTensorHostAllocationUnsupportedV1, BOUNDED_TENSOR_HOST_ALLOCATION_MAX_RANK_V1,
};
/// Compressed f16 bounds for 50% memory reduction vs f32 storage.
pub use compressed::{CompressedBounds, CompressionStats};
/// Generic scalar-type wrappers that adapt external types to internal f32 representation.
pub use generic::{BoundedScalar, GenericBounds};
/// Multi-norm linear bounds for Lp-bounded perturbation sets.
pub use multi_norm::MultiNormBounds;
/// Thread-local memory pool for reusing tensor buffers during bound propagation.
pub use pool::{PoolStats, PooledBuffer, TensorPool};
/// Pooled ndarray storage that auto-returns backing buffers on drop.
pub use pooled_array::PooledArray;
/// Directed rounding utilities for sound floating-point bound arithmetic.
pub use rounding::{
    add_down_f32, add_up_f32, cast_f64_to_f32_down, cast_f64_to_f32_up, div_down_f32, div_up_f32,
    mul_down_f32, mul_up_f32, next_down_f32, next_up_f32, shift_down_n_ulps, shift_up_n_ulps,
    sub_down_f32, sub_up_f32,
};
/// Dynamic tensor storage for branch-and-bound domain management (LIFO/FIFO).
pub use tensor_storage::{
    create_tensor_storage, QueueTensorStorage, StackTensorStorage, TensorStorage, TreeTraversal,
};
/// Sliding window extraction (im2col) for image tensors.
pub use unfold::{inplace_unfold, unfold_output_size};
/// Correlation-aware zonotope tensor for tighter bounds in attention-like operations.
pub use zonotope::ZonotopeTensor;
