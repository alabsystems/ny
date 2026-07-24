// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Parallel verification across sequence positions.
//!
//! Re-exports the curated parallel verification helpers used by external
//! verifier consumers for near-linear speedup on sequence models,
//! including both CPU-default and engine-aware entry points. Import
//! [`crate::verify::GemmEngine`] and [`crate::verify::NaiveCpuGemmEngine`] (or
//! the same names from [`crate::prelude`] when `propagate` is enabled) when
//! using the engine-aware helpers.

pub use ny_propagate::{
    verify_parallel, verify_parallel_with_engine, verify_parallel_with_method,
    verify_parallel_with_method_and_engine, ParallelConfig, ParallelVerificationResult,
    ParallelVerifier,
};
