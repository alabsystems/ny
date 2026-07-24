// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GroupNorm layer for bound propagation.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types (`GroupNormLayer`)
//! - [`math`]: Concrete evaluation (`eval`) and Jacobian computation
//! - [`ibp`]: Interval bound propagation and trait wiring
//! - [`crown_scalar`]: Scalar CROWN backward propagation
//! - [`crown_batched`]: Batched CROWN backward propagation
//!
//! # Reference
//!
//! Wu & He, "Group Normalization," ECCV 2018.
//! GroupNorm(x)[c, t] = ny[c] * (x[c, t] - mean_g) / sqrt(var_g + eps) + beta[c]
//!
//! where g = c / (C / num_groups) is the group index, and mean_g/var_g are computed
//! over all channels in group g and all spatial/time positions.
//!
//! GroupNorm generalizes:
//! - InstanceNorm: num_groups = C (each channel is its own group)
//! - LayerNorm: num_groups = 1 (all channels in one group)
//!
//! Used in: Demucs DConv sub-layers (dilated Conv1d + GroupNorm + GELU).
//! Part of #3205.

mod crown_batched;
mod crown_scalar;
mod ibp;
mod ibp_forward;
mod math;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_crown_ibpvalidated;
pub mod types;

pub use types::GroupNormLayer;
