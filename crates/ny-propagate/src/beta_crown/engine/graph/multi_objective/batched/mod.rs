// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-batched domain processing for single-objective and multi-objective
//! graph BaB verification.
//!
//! Split from the original `batched.rs` monolith (1029 LOC) into focused
//! modules per the multi-objective batching design.

mod batched_dense_specs;
mod batched_multi;
mod batched_single;
pub(super) mod children;
mod kfsb_multi;
#[cfg(test)]
mod tests;
