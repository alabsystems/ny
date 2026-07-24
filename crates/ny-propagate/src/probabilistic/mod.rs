// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Probabilistic bounds via Monte Carlo sampling with CROWN certificates.
//!
//! Phase 1: Monte Carlo wrapper that samples inputs from a distribution,
//! evaluates the network, and provides statistical bounds with confidence
//! intervals. CROWN bounds serve as sound over-approximations that the
//! empirical bounds must lie within.
//!
//! Phase 2: Concentration inequalities (Hoeffding, McDiarmid) for non-asymptotic
//! probabilistic certificates without distribution assumptions.
//!
//! Phase 3: Distributional propagation (Boetius et al., ICML 2025)
//! for distribution-aware bound propagation through CROWN linear relaxations.
//!
//! Part of #3921.

use ndarray::ArrayD;
use ny_core::{NyError, Result};

pub mod branch_and_bound;
pub mod cgf;
pub mod concentration;
pub mod distributional;
pub mod distributional_alpha;
pub mod moments;
pub mod monte_carlo;

/// Validate that all elements of an f32 array are finite (not NaN or Inf).
pub(crate) fn validate_finite_f32(arr: &ArrayD<f32>, name: &str) -> Result<()> {
    if arr.iter().any(|v| !v.is_finite()) {
        return Err(NyError::InvalidSpec(format!(
            "{name} contains NaN or Inf — numerical optimization diverged"
        )));
    }
    Ok(())
}

/// Validate that all elements of an f64 array are finite (not NaN or Inf).
pub(crate) fn validate_finite_f64(arr: &ArrayD<f64>, name: &str) -> Result<()> {
    if arr.iter().any(|v| !v.is_finite()) {
        return Err(NyError::InvalidSpec(format!(
            "{name} contains NaN or Inf — numerical optimization diverged"
        )));
    }
    Ok(())
}
