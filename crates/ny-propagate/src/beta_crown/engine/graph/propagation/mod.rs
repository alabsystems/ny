// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod batched;
mod beta;
mod constraints;
pub(in crate::beta_crown::engine::graph) mod emptiness_slack;
pub(crate) mod warm_frontier;
mod warm_start_guard;
/// #la-warm-frontier tests name the guard's verdict to pick the seeds it refuses.
#[cfg(test)]
pub(in crate::beta_crown::engine::graph) use warm_start_guard::la_warm_start_seed_is_complete as warm_start_guard_for_tests;

// Re-export for graph-level access (used by gpu_bab.rs).
pub(in crate::beta_crown::engine::graph) use batched::{
    BatchedBackwardContext, BatchedBackwardResult,
};
