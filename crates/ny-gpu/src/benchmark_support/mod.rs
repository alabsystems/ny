// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared benchmark support for GPU CROWN backward tests and Criterion benches.
//!
//! This module is `#[doc(hidden)]` — it is not part of the public API contract.
//! It exists so that Criterion bench binaries can `use ny_gpu::benchmark_support::*`
//! without fragile `#[path = "..."]` directives.

pub mod crown_backward_cases;
pub mod crown_backward_measurements;
pub mod crown_backward_profiles;
pub mod crown_backward_workloads;

#[cfg(all(test, feature = "gpu-tests"))]
mod crown_backward_cases_tests;
#[cfg(test)]
mod crown_backward_measurements_tests;
