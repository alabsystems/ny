// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Accelerated bound propagation operations using SIMD + Rayon parallelization.
//!
//! This module provides the `AcceleratedBoundPropagation` trait and its CPU
//! implementation via `AcceleratedDevice`. Standalone computation kernels
//! (linear IBP, matmul IBP, per-position CROWN) are also exposed for use
//! by other device implementations (e.g., `WgpuDevice`).

mod attention;
mod crown_parallel;
mod kernels;

pub use crown_parallel::{
    crown_per_position_parallel, crown_per_position_parallel_with_engine,
    crown_per_position_sequential_with_engine,
};
pub use kernels::{linear_ibp_parallel, matmul_ibp_parallel};

use ndarray::{Array1, Array2};
use ny_core::{checked_shape_product, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use ny_propagate::GraphNetwork;

/// Trait for accelerated bound propagation operations.
pub trait AcceleratedBoundPropagation {
    /// IBP through a linear layer: y = Wx + b
    ///
    /// For interval input [l, u], computes:
    /// - lower: W_pos @ l + W_neg @ u + b
    /// - upper: W_pos @ u + W_neg @ l + b
    ///
    /// where W_pos = max(W, 0), W_neg = min(W, 0)
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor>;

    /// IBP through batched matrix multiplication.
    ///
    /// Supports N-D batch dimensions for transformer attention patterns.
    fn matmul_ibp(&self, input_a: &BoundedTensor, input_b: &BoundedTensor)
        -> Result<BoundedTensor>;

    /// Per-position CROWN using parallel execution.
    ///
    /// For N-D input [...batch_dims..., features], runs CROWN independently
    /// on each position in parallel. This provides significant speedup for
    /// transformer verification where positions are independent.
    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor>;
}

/// Accelerated device using SIMD + Rayon parallelization.
#[derive(Default)]
pub struct AcceleratedDevice;

impl AcceleratedDevice {
    pub fn new() -> Self {
        Self
    }
}

impl AcceleratedBoundPropagation for AcceleratedDevice {
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor> {
        let in_features = weight.ncols();
        let out_features = weight.nrows();

        // Validate input shape
        let shape = input.shape();
        if shape.is_empty() || shape[shape.len() - 1] != in_features {
            return Err(ny_core::NyError::shape_mismatch(
                vec![in_features],
                shape.to_vec(),
            ));
        }

        let batch_dims = &shape[..shape.len() - 1];
        let batch_size: usize = checked_shape_product(batch_dims).ok_or_else(|| {
            ny_core::NyError::InvalidSpec(format!(
                "AcceleratedDevice linear_ibp: batch dims {batch_dims:?} overflow usize",
            ))
        })?;

        debug!(
            "AcceleratedDevice linear_ibp: batch={}, in={}, out={}",
            batch_size, in_features, out_features
        );

        // Use parallel implementation
        linear_ibp_parallel(input, weight, bias, batch_size, in_features, out_features)
    }

    fn matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        matmul_ibp_parallel(input_a, input_b)
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        crown_per_position_parallel(graph, input)
    }
}
