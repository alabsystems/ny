// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! RMSNorm layer for bound propagation.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types (`RmsNormLayer`)
//! - [`math`]: Concrete evaluation (`eval`) and Jacobian computation
//! - [`ibp`]: Interval bound propagation and trait wiring
//! - [`crown_scalar`]: Scalar CROWN backward propagation
//! - [`crown_batched`]: Batched CROWN backward propagation
//!
//! # Reference
//!
//! Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
//! RMSNorm(x) = ny * x / sqrt(mean(x^2) + eps)
//! No mean subtraction (unlike LayerNorm), no beta offset.

mod crown_batched;
mod crown_scalar;
mod ibp;
mod math;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_crown;
#[cfg(test)]
mod tests_ibpvalidated;
#[cfg(test)]
mod tests_multi_dim;
pub mod types;

pub use types::RmsNormLayer;
