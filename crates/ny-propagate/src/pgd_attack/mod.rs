// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD (Projected Gradient Descent) attack for finding counterexamples.
//!
//! This module implements adversarial attacks to find inputs that violate
//! verification properties. When verification is inconclusive, we can try to
//! find a concrete counterexample that proves the property is violated.
//!
//! ## Algorithm
//!
//! 1. **Random Initialization**: Sample random points within input bounds
//! 2. **Gradient Estimation**: Use SPSA to estimate gradients without backprop
//! 3. **Gradient Step**: Move toward property violation
//! 4. **Projection**: Clip to input bounds
//! 5. **Repeat**: Multiple restarts for robustness
//!
//! ## References
//!
//! - α,β-CROWN uses PGD with 10000 restarts for ACAS-Xu benchmarks
//! - SPSA: Spall, J.C. (1992). "Multivariate Stochastic Approximation Using a
//!   Simultaneous Perturbation Gradient Approximation"

mod adam_state;
mod attack_conjunctive_greater_eq;
mod attack_conjunctive_less_eq;
mod attack_difference;
mod attack_disjunctive;
mod attack_disjunctive_greater_eq;
mod attack_disjunctive_less_eq;
mod attacker;
mod config;
pub mod gama;
pub(crate) mod optimizer;
mod result;

pub use attacker::PgdAttacker;
pub use config::{PgdConfig, PgdInitialization, GAMA_LAMBDA_DEFAULT};
pub use optimizer::{
    auto_alpha, project_to_bounds, project_to_bounds_in_place, AdamClippingParams, PgdAlphaMode,
    PgdOptimizer, PgdStepState,
};
pub use result::PgdResult;

#[cfg(test)]
mod tests;
