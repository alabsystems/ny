// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! INVPROP: Output Constraint Backward Propagation
//!
//! This module implements INVPROP (Kotha et al., 2023), which propagates output
//! constraints backward through the network to tighten intermediate bounds.
//!
//! # Overview
//!
//! INVPROP introduces nonnegative dual variables ("gammas") that dualize output
//! constraints `A*y <= rhs`. The backward pass carries the dualized constraint
//! terms, but the optimization step for the gammas is NOT implemented yet: they
//! stay at their zero initialization, so enabling INVPROP currently does not
//! tighten any bounds.
//!
//! # Key Components
//!
//! - [`InvpropConfig`]: Configuration parameters for INVPROP
//! - [`OutputConstraints`]: Matrix representation of output constraints
//! - [`LayerGammas`]: Per-layer dual variables for constraint dualization
//! - [`InvpropState`]: State for INVPROP optimization across the network
//!
//! # Reference
//!
//! Kotha et al., "Provably Computing the Preimage of Deep Neural Networks",
//! arXiv:2302.01404 (NeurIPS 2023)

mod config;
mod constraints;
mod gammas;
pub(crate) mod split_lift;

pub use config::InvpropConfig;
pub use constraints::OutputConstraints;
pub use gammas::{InvpropState, LayerGammas};

/// Reserved [`InvpropState::layer_gammas`] key for the **output identity seed**
/// duals (the output-node-only assume-violation channel).
///
/// The seed gammas have "neuron" dimension `= output_dim` (one dual per output
/// coordinate per constraint). Folding the raw, output-indexed constraint matrix
/// `C` is only dimensionally valid at this seed, which is why the shipped
/// (output-node-only) path keys its duals here rather than per layer.
pub const INVPROP_OUTPUT_SEED: &str = "__invprop_output_seed__";

#[cfg(test)]
mod tests;
