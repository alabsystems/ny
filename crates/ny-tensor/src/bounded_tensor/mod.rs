// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bounded tensor implementation split into core definitions and ops.

mod core;
/// Double-precision bounded tensor for soundness-critical f64 propagation.
pub(crate) mod float64;
mod l2_constraint;
mod ops;

#[cfg(test)]
mod proptest_soundness;
#[cfg(test)]
mod tests;

/// Core interval-bounded tensor type used across bound propagation layers.
pub use core::BoundedTensor;
/// Shared inverted-bounds repair strategy for propagation and readback code. Part of #3307.
pub use core::InversionRepair;
/// Strategy for automatic NaN/Inf repair at the type boundary. Part of #3423.
pub use core::RepairStrategy;
/// Shared inverted-bounds repair helpers. Part of #3307.
pub use core::{repair_inverted_bounds, repair_inverted_bounds_nd};
/// Double-precision bounded tensor for f64 propagation (soundnessbench, sat_relu).
pub use float64::BoundedTensor64;
/// Optional per-normalization-slice Euclidean-ball annotation enabling exact
/// Cauchy–Schwarz tightening at the immediately-downstream `Linear`.
pub use l2_constraint::L2Constraint;
