// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm layer for bound propagation.
//!
//! `LayerNormCrownMode::IbpValidated` uses the shared decomposed primitive
//! chain with rowwise fallback to fused LayerNorm IBP, matching the
//! alpha-beta-CROWN-style complex-node decomposition strategy. `Sampling`
//! remains an explicit heuristic mode.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types (`LayerNormCrownMode`, `LayerNormMode`, `LayerNormLayer`)
//! - [`math`]: Concrete evaluation (`eval`) and Jacobian computation
//! - [`ibp`]: Interval bound propagation and trait wiring
//! - [`crown_scalar`]: Scalar CROWN backward propagation
//! - [`crown_batched`]: Batched CROWN backward propagation
//!
//! # Reference
//!
//! Based on alpha-beta-CROWN: `auto_LiRPA/operators/normalization.py`

mod crown_batched;
mod crown_scalar;
mod ibp;
mod math;
mod sampling_low_rank;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_inv_n;
#[cfg(test)]
mod tests_multi_dim;
pub mod types;

pub use types::{LayerNormCrownMode, LayerNormLayer, LayerNormMode};
