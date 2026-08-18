// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backward propagation methods for β-CROWN verifier.
//!
//! This module contains gradient computation and layer backward propagation
//! with α and β parameters for the Lagrangian optimization in β-CROWN.
//!
//! ## Submodules
//!
//! - [`shape_inference`]: Conv input spatial dimension inference from flat shapes
//! - [`relu_backward`]: Production ReLU backward with α and β
//! - [`layer_dispatch`]: Layer-type dispatch for backward propagation
//! - [`legacy`]: Test-only β-only backward passes and relaxation recording

mod layer_dispatch;
#[cfg(test)]
mod legacy;
mod relu_backward;
mod shape_inference;

#[cfg(test)]
pub(super) use legacy::ReluLowerRelaxation;

// Re-exported for tests module access via `super::BetaCrownVerifier`.
#[cfg(test)]
pub(super) use super::BetaCrownVerifier;

#[cfg(test)]
mod tests;
