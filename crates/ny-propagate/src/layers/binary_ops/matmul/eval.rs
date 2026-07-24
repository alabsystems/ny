// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array2;

use super::{MatMulLayer, Result};

impl MatMulLayer {
    /// Evaluate MatMul at a concrete point: C = A @ B (or A @ B^T).
    /// For 2D inputs only.
    pub fn eval(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let b_for_matmul = if self.transpose_b {
            b.t().to_owned()
        } else {
            b.clone()
        };

        let mut result = a.dot(&b_for_matmul);

        if let Some(scale) = self.scale {
            result.mapv_inplace(|v| v * scale);
        }

        Ok(result)
    }

    /// Compute the Jacobian of C = A @ B w.r.t. A at a fixed B value.
    ///
    /// `C[i,j]` = Σ_l `A[i,l]` * `B[l,j]` (or `B[j,l]` if transpose_b)
    /// `∂C[i,j]/∂A[p,q]` = δ_{ip} * `B[q,j]` (or `B[j,q]` if transpose_b)
    ///
    /// Returns a matrix J of shape (m*n, m*k) where:
    /// - C is flattened row-major to length m*n
    /// - A is flattened row-major to length m*k
    pub fn jacobian_wrt_a(&self, b: &Array2<f32>) -> Array2<f32> {
        let (_k, _n) = if self.transpose_b {
            (b.ncols(), b.nrows())
        } else {
            (b.nrows(), b.ncols())
        };

        // We need to know m (rows of A), but we don't have A here
        // The Jacobian shape depends on A's shape too
        // For now, we'll compute the transformation directly in the propagation method
        // This helper just returns B in the right orientation for later use
        let b_effective = if self.transpose_b {
            b.t().to_owned()
        } else {
            b.clone()
        };

        // Return B^T which is used in the backward transformation
        // (with optional scaling)
        let mut result = b_effective.t().to_owned();
        if let Some(scale) = self.scale {
            result.mapv_inplace(|v| v * scale);
        }
        result
    }

    /// Compute the Jacobian of C = A @ B w.r.t. B at a fixed A value.
    ///
    /// `C[i,j]` = Σ_l `A[i,l]` * `B[l,j]`
    /// `∂C[i,j]/∂B[p,q]` = `A[i,p]` * δ_{jq}
    ///
    /// Returns a matrix that can be used in backward propagation.
    pub fn jacobian_wrt_b(&self, a: &Array2<f32>) -> Array2<f32> {
        // Return A^T which is used in the backward transformation
        let mut result = a.t().to_owned();
        if let Some(scale) = self.scale {
            result.mapv_inplace(|v| v * scale);
        }
        result
    }
}
