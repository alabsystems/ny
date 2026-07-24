// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Streaming / memory-efficient verification of large networks.
//!
//! Re-exports the curated streaming verification surface used by external
//! consumers to verify large transformer networks under tight memory budgets.
//! [`StreamingVerifier`] applies gradient checkpointing during CROWN
//! propagation: instead of retaining bounds at every layer (O(L*N) memory), it
//! stores [`CheckpointedBounds`] at intervals controlled by [`StreamingConfig`]
//! and recomputes forward on the backward pass, trading compute for memory.
//! Use [`estimate_memory_savings`] to size the trade-off before verifying.

pub use ny_propagate::{
    estimate_memory_savings, CheckpointedBounds, StreamingConfig, StreamingVerifier,
};
