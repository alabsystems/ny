// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! InstanceNorm1d layer for bound propagation.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types (`InstanceNorm1dLayer`)
//! - [`math`]: Concrete evaluation (`eval`) and Jacobian computation
//! - [`ibp`]: Interval bound propagation and trait wiring
//! - [`crown_scalar`]: Scalar CROWN backward propagation
//! - [`crown_batched`]: Batched CROWN backward propagation
//!
//! # Reference
//!
//! Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization," 2016.
//! InstanceNorm1d(x)[c, t] = ny[c] * (x[c, t] - mean(x[c, :])) / sqrt(var(x[c, :]) + eps) + beta[c]
//!
//! Unlike LayerNorm (which normalizes across the feature dimension), InstanceNorm normalizes
//! across the time/spatial dimension independently for each channel.
//!
//! Used in: avoice (AdaIN, Snake+InstanceNorm pipelines), style transfer.

mod crown_batched;
mod crown_scalar;
mod ibp;
mod math;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_crown;
pub mod types;

pub use types::InstanceNorm1dLayer;
