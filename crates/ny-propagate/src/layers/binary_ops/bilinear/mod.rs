// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::matmul::MatMulLayer;
use crate::layers::activations::validate::validate_finite;

mod batched;
mod linear;
mod mccormick;
mod nd_compose;
mod relaxation;
#[cfg(test)]
mod tests;

use mccormick::interpolated_mccormick;
pub(crate) use relaxation::BilinearRelaxation;

/// Bilinear CROWN layer for N-D CROWN composition through attention Q@K^T.
///
/// This layer enables proper McCormick envelope composition via
/// `BilinearRelaxation::compose_backward_broadcast` for attention matmul operations.
/// Unlike MatMulLayer which computes McCormick bounds but doesn't compose them with
/// downstream bounds, BilinearCrownLayer maintains the full N-D CROWN chain through
/// bilinear operations.
///
/// # Background
///
/// For z = Q @ K^T where Q ∈ [Q_l, Q_u] and K ∈ [K_l, K_u], McCormick gives:
/// - z ≥ Q_l * K + K_l * Q - Q_l * K_l  (lower envelope)
/// - z ≤ Q_u * K + K_l * Q - Q_u * K_l  (upper envelope, one of several)
///
/// When propagating downstream bounds y = A_down @ z + b_down through this
/// bilinear operation, we need broadcast composition to properly handle:
/// - Shape transformation from per-position [b,h,s,s,s] to flattened [b,h,flat,q]
/// - Bias term composition with McCormick envelope constants
///
/// # Usage
///
/// Used in attention graph construction (native/helpers.rs, native/build/transformer.rs) instead of MatMulLayer
/// for Q@K^T operations when tight bounds are needed.
#[derive(Debug, Clone)]
pub struct BilinearCrownLayer {
    /// Whether to transpose the second input (K^T pattern).
    pub(crate) transpose_b: bool,
    /// Optional scaling factor (e.g., 1/sqrt(d_k) for attention).
    /// `pub(crate)` to prevent bypassing `try_new` finite validation (#4307).
    pub(crate) scale: Option<f32>,
}

impl BilinearCrownLayer {
    /// Validate and create a new BilinearCrown layer.
    pub fn try_new(transpose_b: bool, scale: Option<f32>) -> Result<Self> {
        Ok(Self {
            transpose_b,
            scale: scale
                .map(|scale| validate_finite(scale, "BilinearCrownLayer", "scale"))
                .transpose()?,
        })
    }

    /// Create a new BilinearCrown layer.
    pub fn new(transpose_b: bool, scale: Option<f32>) -> Self {
        Self::try_new(transpose_b, scale)
            .expect("invariant: BilinearCrownLayer::new requires finite scale")
    }

    /// Whether the second input is transposed.
    pub fn transpose_b(&self) -> bool {
        self.transpose_b
    }

    /// Return the optional scale factor.
    pub fn scale(&self) -> Option<f32> {
        self.scale
    }

    /// IBP propagation delegates to MatMulLayer (same interval arithmetic).
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let matmul = MatMulLayer::new(self.transpose_b, self.scale);
        matmul.propagate_ibp_binary(input_a, input_b)
    }
}
