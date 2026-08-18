// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Normalization layers for bound propagation.
//!
//! This module provides LayerNorm and BatchNorm implementations with IBP and CROWN
//! bound propagation support.

mod adain;
mod batch_norm;
pub(crate) mod crown_batched_common;
pub(crate) mod crown_common;
pub(crate) mod decomposed;
mod group_norm;
mod instance_norm;
pub(crate) mod layer_norm;
pub(crate) mod math_common;
mod rms_norm;
pub(crate) mod trait_norm;
pub(crate) mod validate;

/// Smallest authored epsilon supported by the fused normalization layers.
///
/// Values below this threshold are rejected, never silently rounded upward:
/// changing epsilon changes the represented function and can change a
/// verification verdict at a tight property boundary.
pub const NORMALIZATION_MIN_EPS: f32 = 1e-12;

// Re-export all public types at module root for backward compatibility
pub use adain::AdaIN1dLayer;
pub use batch_norm::{BatchNormChannelAxisHint, BatchNormLayer};
pub use group_norm::GroupNormLayer;
pub use instance_norm::InstanceNorm1dLayer;
pub use layer_norm::{LayerNormCrownMode, LayerNormLayer, LayerNormMode};
pub use rms_norm::RmsNormLayer;
