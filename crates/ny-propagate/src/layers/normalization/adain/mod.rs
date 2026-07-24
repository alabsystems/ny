// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AdaIN1d (Adaptive Instance Normalization) layer for bound propagation.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types (`AdaIN1dLayer`)
//! - [`math`]: Concrete evaluation
//! - [`ibp`]: Interval bound propagation and trait wiring
//! - [`crown_scalar`]: Scalar CROWN backward propagation
//! - [`crown_batched`]: Batched CROWN backward propagation
//!
//! # Reference
//!
//! Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance
//! Normalization," ICCV 2017.
//!
//! AdaIN(x) = style_gamma * InstanceNorm(x) + style_beta
//!
//! where style_gamma and style_beta are per-channel parameters from a style network.
//! At inference with fixed style, these are constants extracted from ONNX initializers.
//!
//! Used in: avoice K3 (AdaIN vocoder), K4 (Snake+AdaIN pipeline), style transfer.

mod crown_batched;
mod crown_scalar;
mod crown_ternary;
mod ibp;
mod math;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_crown;
#[cfg(test)]
mod tests_variable_style;
pub mod types;

pub use types::AdaIN1dLayer;
