// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MatMul layer — bounded matrix multiplication (A @ B or A @ B^T).
//!
//! Split into focused submodules:
//! - `shape`: dimension parsing and batch-index helpers
//! - `ibp_standard`: standard interval arithmetic IBP
//! - `ibp_economic`: memory-efficient economic IBP
//! - `crown_dense`: non-batched CROWN backward propagation
//! - `crown_batched`: batched CROWN backward propagation
//! - `eval`: concrete evaluation / Jacobian
//! - `helpers`: utility functions (scale, non-finite repair, finite checks)
//!
//! McCormick guard validation is in `binary_ops::validate_mccormick_inputs`.

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

// Re-exports for submodules (accessed via `super::`).
use super::mul::{select_mccormick_plane, BoundDir};
use crate::layers::activations::validate::validate_finite;
use crate::{BatchedLinearBounds, LinearBounds};

// Re-export shape helpers for sibling modules (BilinearCrownLayer alpha support, #3287).
pub(super) use shape::{decode_batch_index_into_buf, parse_matmul_dims};

mod crown_batched;
mod crown_dense;
mod eval;
mod helpers;
mod ibp_economic;
mod ibp_standard;
#[cfg(test)]
mod rounding_tests;
mod shape;
#[cfg(test)]
mod tests;

/// IBP mode for MatMul when both inputs are perturbed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatMulIbpMode {
    /// Standard interval arithmetic (tightest, but may expand large tensors).
    #[default]
    Standard,
    /// Economic IBP (memory-efficient, potentially looser).
    /// Only used when both inputs are perturbed and all bounds are finite.
    /// Falls back to Standard otherwise.
    Economic,
}

/// Bounded matrix multiplication layer for operations like Q @ K^T in attention.
///
/// Unlike LinearLayer which has fixed weights, MatMulLayer multiplies two
/// bounded tensor inputs. This is used for attention score computation
/// and attention-value multiplication.
///
/// For C = A @ B where A ∈ \[A_l, A_u\] and B ∈ \[B_l, B_u\]:
/// Each element `c[i,k]` = sum_j(`a[i,j]` * `b[j,k]`) is bounded using interval arithmetic.
#[derive(Debug, Clone)]
pub struct MatMulLayer {
    /// Whether to transpose the second input (B^T instead of B).
    /// Used for Q @ K^T attention pattern.
    pub(crate) transpose_b: bool,
    /// Optional scaling factor (e.g., 1/sqrt(d_k) for attention).
    /// `pub(crate)` to prevent bypassing `try_new` finite validation (#4307).
    pub(crate) scale: Option<f32>,
    /// IBP mode selection for MatMul when both operands are perturbed.
    pub(crate) ibp_mode: MatMulIbpMode,
}

impl MatMulLayer {
    /// Validate and create a new MatMul layer.
    pub fn try_new(transpose_b: bool, scale: Option<f32>) -> Result<Self> {
        Ok(Self {
            transpose_b,
            scale: scale
                .map(|scale| validate_finite(scale, "MatMulLayer", "scale"))
                .transpose()?,
            ibp_mode: MatMulIbpMode::Standard,
        })
    }

    /// Create a new MatMul layer.
    pub fn new(transpose_b: bool, scale: Option<f32>) -> Self {
        Self::try_new(transpose_b, scale)
            .expect("invariant: MatMulLayer::new requires finite scale")
    }

    /// Validate and create a new MatMul layer with a specific IBP mode.
    pub fn try_new_with_ibp_mode(
        transpose_b: bool,
        scale: Option<f32>,
        ibp_mode: MatMulIbpMode,
    ) -> Result<Self> {
        let mut layer = Self::try_new(transpose_b, scale)?;
        layer.ibp_mode = ibp_mode;
        Ok(layer)
    }

    /// Create a new MatMul layer with a specific IBP mode.
    pub fn new_with_ibp_mode(
        transpose_b: bool,
        scale: Option<f32>,
        ibp_mode: MatMulIbpMode,
    ) -> Self {
        Self::try_new_with_ibp_mode(transpose_b, scale, ibp_mode)
            .expect("invariant: MatMulLayer::new_with_ibp_mode requires finite scale")
    }

    /// Set the IBP mode on an existing MatMul layer.
    pub fn with_ibp_mode(mut self, ibp_mode: MatMulIbpMode) -> Self {
        self.ibp_mode = ibp_mode;
        self
    }

    /// Whether the second input is transposed.
    pub fn transpose_b(&self) -> bool {
        self.transpose_b
    }

    /// Return the optional scale factor.
    pub fn scale(&self) -> Option<f32> {
        self.scale
    }

    /// Return the IBP mode.
    pub fn ibp_mode(&self) -> MatMulIbpMode {
        self.ibp_mode
    }

    /// Propagate IBP bounds through matrix multiplication of two bounded tensors.
    ///
    /// For A @ B (or A @ B^T if transpose_b), computes interval bounds on the result.
    /// Dispatches to standard or economic IBP based on `ibp_mode` and input properties.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if self.should_use_economic_ibp(input_a, input_b) {
            let dims = parse_matmul_dims(self.transpose_b, input_a.shape(), input_b.shape())?;
            let mut out_shape = dims.batch_dims.clone();
            out_shape.push(dims.m);
            out_shape.push(dims.n);
            return self.propagate_ibp_economic(input_a, input_b, &dims, &out_shape);
        }
        self.propagate_ibp_standard(input_a, input_b)
    }

    fn should_use_economic_ibp(&self, input_a: &BoundedTensor, input_b: &BoundedTensor) -> bool {
        if self.ibp_mode != MatMulIbpMode::Economic {
            return false;
        }
        if !helpers::bounds_all_finite(input_a) || !helpers::bounds_all_finite(input_b) {
            return false;
        }
        helpers::is_perturbed(input_a) && helpers::is_perturbed(input_b)
    }

    // propagate_linear_binary is defined in crown_dense.rs
    // propagate_linear_batched_binary is defined in crown_batched.rs
}
