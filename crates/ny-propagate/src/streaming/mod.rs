// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Streaming computation with gradient checkpointing for memory-efficient verification.
//!
//! This module implements gradient checkpointing to reduce memory usage during CROWN
//! propagation. Instead of storing bounds at every layer (O(L*N)), we store checkpoints
//! at intervals and recompute forward during the backward pass when needed.
//!
//! **Memory-Compute Trade-off:**
//! - Without checkpointing: O(L*N) memory, O(L) compute
//! - With K-interval checkpointing: O(L/K * N) memory, O(L*K) compute
//!
//! For a 100-layer network with K=10, this reduces memory by ~90% at 10x compute cost.
//! Since modern GPUs are compute-bound, this is often a good trade-off.
//!
//! ## Compressed Storage (f16)
//!
//! When `use_f16_checkpoints` is enabled, checkpoints are stored using f16 (half precision)
//! which provides an additional 50% memory reduction on top of checkpointing. This is
//! particularly useful for very large models or memory-constrained environments.
//!
//! **Memory with f16 + checkpointing:**
//! - Original: O(L*N) memory (f32)
//! - Checkpointing only (K=10): O(L/K * N) f32 = ~10% of original
//! - Checkpointing + f16: O(L/K * N/2) = ~5% of original

mod checkpoint;
mod config;
mod memory;
mod verifier;

#[cfg(test)]
mod tests;

pub use checkpoint::CheckpointedBounds;
pub use config::StreamingConfig;
pub use memory::estimate_memory_savings;
pub use verifier::StreamingVerifier;
