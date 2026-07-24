// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::accelerated::AcceleratedBoundPropagation;
use ndarray::{Array1, Array2};
use ny_core::{checked_shape_product, NyError, Result};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::WgpuDevice;

impl AcceleratedBoundPropagation for WgpuDevice {
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
            return Err(NyError::shape_mismatch(vec![in_features], shape.to_vec()));
        }

        let batch_dims = &shape[..shape.len() - 1];
        let batch_size: usize = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "WgpuDevice linear_ibp: batch dims {batch_dims:?} overflow usize",
            ))
        })?;

        debug!(
            "WgpuDevice linear_ibp: batch={}, in={}, out={}",
            batch_size, in_features, out_features
        );

        // Wrap GPU work in an error scope: a wgpu validation/internal/OOM error
        // becomes Err (caller falls back to CPU) instead of aborting the process
        // via wgpu's panicking uncaptured-error handler (#live bug).
        self.run_gpu_checked("linear_ibp", || {
            self.execute_linear_ibp(input, weight, bias, batch_size, in_features, out_features)
        })
    }

    fn matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let shape_a = input_a.shape();
        let shape_b = input_b.shape();

        if shape_a.len() < 2 || shape_b.len() < 2 {
            return Err(NyError::shape_mismatch(
                vec![2],
                vec![shape_a.len().min(shape_b.len())],
            ));
        }

        // Get matrix dimensions
        let m = shape_a[shape_a.len() - 2]; // rows of A
        let k = shape_a[shape_a.len() - 1]; // cols of A = rows of B
        let n = shape_b[shape_b.len() - 1]; // cols of B

        // Verify inner dimensions match
        if shape_b[shape_b.len() - 2] != k {
            return Err(NyError::shape_mismatch(
                vec![k],
                vec![shape_b[shape_b.len() - 2]],
            ));
        }

        // Compute batch dimensions
        let batch_dims_a = &shape_a[..shape_a.len() - 2];
        let batch_dims_b = &shape_b[..shape_b.len() - 2];

        // Batch dims should match (simplified - full broadcasting not implemented)
        if batch_dims_a != batch_dims_b {
            return Err(NyError::shape_mismatch(
                batch_dims_a.to_vec(),
                batch_dims_b.to_vec(),
            ));
        }

        let batch_size: usize = checked_shape_product(batch_dims_a).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "WgpuDevice matmul_ibp: batch dims {batch_dims_a:?} overflow usize",
            ))
        })?;

        debug!(
            "WgpuDevice matmul_ibp: batch={}, m={}, k={}, n={}",
            batch_size, m, k, n
        );

        self.run_gpu_checked("matmul_ibp", || {
            self.execute_matmul_ibp(input_a, input_b, batch_size, m, k, n, batch_dims_a)
        })
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // GPU engines (wgpu) can't safely participate in Rayon parallel CROWN due to internal
        // buffer reuse and readback synchronization. Instead, run per-position CROWN
        // sequentially while accelerating GEMM via the GPU.
        debug!(
            "WgpuDevice crown_per_position_parallel: using GPU-accelerated sequential per-position CROWN"
        );
        crate::accelerated::crown_per_position_sequential_with_engine(graph, input, Some(self))
    }
}

impl WgpuDevice {
    pub(crate) fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor> {
        <Self as AcceleratedBoundPropagation>::linear_ibp(self, input, weight, bias)
    }

    pub(crate) fn matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        <Self as AcceleratedBoundPropagation>::matmul_ibp(self, input_a, input_b)
    }

    pub(crate) fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        <Self as AcceleratedBoundPropagation>::crown_per_position_parallel(self, graph, input)
    }
}
