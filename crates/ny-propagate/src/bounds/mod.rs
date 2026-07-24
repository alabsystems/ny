// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Linear bounds representation for CROWN-style bound propagation.
//!
//! This module contains:
//! - `LinearBounds`: 2D linear bounds (flattened) for basic CROWN
//! - `BatchedLinearBounds`: N-D batched bounds for transformer verification
//! - `AlphaCrownConfig`: Configuration for α-CROWN optimization
//! - `AlphaState`: Learnable α parameters for unstable ReLU neurons

mod alpha;
mod alpha_config;
pub(crate) mod alpha_reciprocal;
mod alpha_s_shaped;
mod alpha_s_shaped_update;
mod alpha_sqrt;
mod batched;
mod batched_f64;
mod concretize;
pub(crate) use concretize::concretize_row_directed;
pub mod facet_bank;
mod linear;
/// Double-precision CROWN linear bounds for f64 propagation.
pub mod linear_f64;
pub(crate) mod patches;
pub(crate) mod patches_batched;
pub(crate) mod patches_ops;
pub(crate) mod safe_math;

#[cfg(test)]
mod alpha_s_shaped_tests;
#[cfg(test)]
mod tests;

pub use alpha::{
    AdamParams, AlphaCrownConfig, AlphaCrownIntermediate, AlphaSpecEarlyExit, AlphaState,
    GradientMethod, GraphAlphaCrownIntermediate, GraphAlphaState, MultiSpecKeep, Optimizer,
};
pub(crate) use alpha_s_shaped::{MonotoneSShapedAlpha, MonotoneSShapedPathAlpha};
pub(crate) use alpha_s_shaped_update::MonotoneSShapedGradients;
pub(crate) use alpha_sqrt::SqrtGradients;
pub use batched::BatchedLinearBounds;
pub(crate) use batched_f64::BatchedLinearBounds64;
pub use facet_bank::{
    FacetBank, FacetBankBound, FacetBankSearchConfig, LowerAffineCertificate,
    FACET_BANK_DEFAULT_DYADIC_BITS, FACET_BANK_MAX_PLANES,
};
pub use linear::LinearBounds;
// NY_SLACK_PROBE: dark f32-soundness-slack accumulator readers (default-off).
pub use linear::{slack_probe_enabled, slack_probe_take};
pub use linear_f64::LinearBounds64;
/// NaN-propagating comparisons (re-exported from ny-core) and safe bound arithmetic.
/// Only items with crate-internal consumers are re-exported here (#3240).
pub(crate) use safe_math::{
    nan_propagating_max, nan_propagating_max_zero, nan_propagating_min, nan_propagating_min_zero,
};
pub use safe_math::{safe_mul_for_bounds, safe_mul_for_bounds_f64};

// Test-only re-exports: these functions have no production callers but are exercised
// by unit tests that import via `crate::bounds::*`. Gated behind #[cfg(test)] so they
// don't leak into the public API (#3240).
#[cfg(test)]
pub(crate) use batched::{
    batched_interval_matvec, batched_interval_matvec_checked, batched_matvec,
};
pub use safe_math::{
    interval_mul_for_bounds, safe_add_for_bounds_with_polarity, safe_add_lower_for_bounds,
    safe_add_upper_for_bounds, safe_mul_pair_for_bounds,
};
#[cfg(test)]
pub use safe_math::{safe_add_for_bounds, safe_array_add, safe_array_add_checked};
