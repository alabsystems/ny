// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared decomposed normalization CROWN helpers.
//!
//! This module hosts the reusable primitive-chain normalization backward paths
//! that are shared between block-wise graph CROWN and the layer-level
//! `LayerNormLayer` / `RmsNormLayer` surfaces.
//!
//! Part of #2077, #318, #3447, #3821, #3830.

mod bilinear;
mod common;
mod grouped_centered;
mod instance_norm;
mod layernorm;
mod rmsnorm;
#[cfg(test)]
mod tests_bilinear;
#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_grouped_centered;
#[cfg(test)]
mod tests_instance_norm;
#[cfg(test)]
mod tests_layernorm;
#[cfg(test)]
mod tests_rmsnorm;
#[cfg(test)]
mod tests_support;
#[cfg(test)]
mod tests_variance;
mod variance_chain;

use crate::bounds::BatchedLinearBounds;

pub(crate) use common::{
    batched_bounds_to_scalar, batched_bounds_to_scalar_multi_dim, finalize_decomposed_norm_bounds,
    scalar_bounds_to_batched, scalar_bounds_to_batched_multi_dim, validate_norm_against_fused_ibp,
    DecomposedNormFinalizeMetadata,
};
pub(crate) use grouped_centered::decomposed_grouped_centered_crown_backward;
pub(crate) use instance_norm::{
    decomposed_instance_norm_crown_backward,
    decomposed_instance_norm_crown_backward_channel_batched,
};
pub(crate) use layernorm::decomposed_norm_crown_backward;
pub(crate) use rmsnorm::{
    decomposed_rms_norm_crown_backward, decomposed_rms_norm_crown_backward_with_override,
    InvRmsOverride,
};

/// Row-validation counts from decomposed normalization CROWN backward.
///
/// Tracks how many rows collapsed to fused LayerNorm IBP because the decomposed
/// CROWN result was looser than the fused baseline. Part of #318, #2077.
#[derive(Debug, Clone, Default)]
pub(crate) struct RowValidationCounts {
    pub fallback_rows: usize,
    pub total_rows: usize,
}

/// Result from decomposed normalization CROWN backward including validation stats.
pub(crate) struct DecomposedNormBackwardResult {
    pub bounds: BatchedLinearBounds,
    pub validation: RowValidationCounts,
}
