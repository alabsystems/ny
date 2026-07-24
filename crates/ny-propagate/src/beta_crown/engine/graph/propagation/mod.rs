// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod batched;
mod beta;
mod constraints;

// Re-export for graph-level access (used by gpu_bab.rs).
pub(in crate::beta_crown::engine::graph) use batched::{
    BatchedBackwardContext, BatchedBackwardResult,
};
