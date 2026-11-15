// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Allocation policy for the ONE full-k, column-major AW/absolute-sum body.
//! Implementations own their buffers; the kernel only borrows fixed views.
//! A storage policy must account for hidden GEMM scratch or refuse that call.

use faer::{Mat, MatMut, MatRef};
use ndarray::Array2;
use ny_core::Result;

pub(crate) trait AwStorage {
    type Matrix;
    type Output;

    fn matrix_ref(matrix: &Self::Matrix) -> MatRef<'_, f64>;
    fn matrix_mut(matrix: &mut Self::Matrix) -> MatMut<'_, f64>;
    fn output_slice(output: &mut Self::Output) -> &mut [f64];
    fn matrix_from_fn(
        &mut self,
        rows: usize,
        cols: usize,
        getter: impl FnMut(usize, usize) -> f64,
    ) -> Result<Self::Matrix>;
    fn matrix_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Matrix>;
    fn output_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Output>;

    /// Called before ANY kernel buffer is allocated or filled. Includes the
    /// last, possibly ragged row block and whether an external cache is used.
    fn admit_products(
        &mut self,
        _m: usize,
        _k: usize,
        _p: usize,
        _rows_per_chunk: usize,
        _cached: bool,
    ) -> Result<()> {
        Ok(())
    }
    fn before_product(&mut self, _rows: usize, _k: usize, _cols: usize) -> Result<()> {
        Ok(())
    }
    fn before_copy_pair(&mut self, _rows: usize, _cols: usize) -> Result<()> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }

    /// The ordinary policy keeps the original single full-width dispatch.
    /// A policy overriding this operation must state its arithmetic contract
    /// and account for every indivisible call before entering that call.
    fn product(&mut self, lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> Result<Self::Matrix> {
        self.before_product(lhs.nrows(), lhs.ncols(), rhs.ncols())?;
        let mut dst = self.matrix_zeros(lhs.nrows(), rhs.ncols())?;
        crate::faer_parallelism::mat_mul_f64_views_into(Self::matrix_mut(&mut dst), lhs, rhs);
        self.checkpoint()?;
        Ok(dst)
    }
}

/// Original owning allocation operations. The normal caller keeps its cache,
/// output layouts, zero fills, full-k reduction and faer parallelism policy.
pub(super) struct OwnedAwStorage;

impl AwStorage for OwnedAwStorage {
    type Matrix = Mat<f64>;
    type Output = Array2<f64>;

    fn matrix_ref(matrix: &Self::Matrix) -> MatRef<'_, f64> {
        matrix.as_ref()
    }
    fn matrix_mut(matrix: &mut Self::Matrix) -> MatMut<'_, f64> {
        matrix.as_mut()
    }
    fn output_slice(output: &mut Self::Output) -> &mut [f64] {
        output
            .as_slice_mut()
            .expect("AW output is freshly allocated row-major contiguous")
    }
    fn matrix_from_fn(
        &mut self,
        rows: usize,
        cols: usize,
        getter: impl FnMut(usize, usize) -> f64,
    ) -> Result<Self::Matrix> {
        Ok(Mat::from_fn(rows, cols, getter))
    }
    fn matrix_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Matrix> {
        Ok(Mat::zeros(rows, cols))
    }
    fn output_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Output> {
        Ok(Array2::zeros((rows, cols)))
    }
}

/// Allocate through the supplied owner, then call the common faer dispatch.
/// This never tiles k, changes destination layout or selects parallelism.
pub(super) fn product<S: AwStorage>(
    storage: &mut S,
    lhs: MatRef<'_, f64>,
    rhs: MatRef<'_, f64>,
) -> Result<S::Matrix> {
    storage.product(lhs, rhs)
}

#[cfg(test)]
#[path = "aw_storage_tests.rs"]
mod tests;
