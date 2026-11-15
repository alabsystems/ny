// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! INVPROP constraint-aware backward propagation routing.
//!
//! When INVPROP is enabled:
//! - Layers in `apply_output_constraints_to` route to INVPROP-specific backward
//! - Output constraints are applied via coefficient augmentation
//! - Infeasibility is detected and tracked
//!
//! # Reference
//!
//! Kotha et al., "Provably Computing the Preimage of Deep Neural Networks",
//! arXiv:2302.01404 (NeurIPS 2023)

mod augment;
mod best_bounds;
#[cfg(test)]
mod tests;

pub(crate) use augment::augment_bounds_with_constraints;
pub(crate) use best_bounds::take_best_bounds;
