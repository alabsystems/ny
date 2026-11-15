// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Output partitioning for the existing full-k AW/absolute-sum body.
//!
//! This policy deliberately permits different reduction bits from full-width
//! faer. Each destination is replaced once by ONE full-k RN-f64 product. The
//! caller must use the existing order-independent coefficient-error enclosure;
//! this is not an interchangeable bit-parity policy for arbitrary GEMMs.
//! No k partition, intermediate sum, scalar dot product or packing cache is
//! introduced. The finite traversal is still disabled.

use super::*;

/// Stack-only plan. Every tile excludes faer's matvec/rank-update paths and
/// costs at most the existing work quantum, inside the audited nano threshold.
#[derive(Clone, Copy, Debug)]
struct OutputPlan {
    rows: usize,
    cols: usize,
    row_limit: usize,
    col_limit: usize,
}

impl OutputPlan {
    fn new(rows: usize, k: usize, cols: usize) -> Result<Self> {
        LeasedAwStorage::product_units(rows, k, cols)?;
        let unsupported = || {
            NyError::UnsupportedOp(
                "bounded tiled AW: no full-k nondegenerate nano partition within work quantum"
                    .into(),
            )
        };
        if !cfg!(target_arch = "aarch64") || rows <= 1 || k <= 1 || cols <= 1 {
            return Err(unsupported());
        }
        // An odd extent needs at least one block of three; even extents can
        // use pairs. This avoids all singleton tails without padding/copying.
        let min_rows = 2 + rows % 2;
        let min_cols = 2 + cols % 2;
        let max_macs = POLL_CHUNK_UNITS / 4;
        let row_limit = rows.min(max_macs / (k * min_cols));
        if row_limit < min_rows {
            return Err(unsupported());
        }
        let col_limit = cols.min(max_macs / (k * row_limit));
        if col_limit < min_cols {
            return Err(unsupported());
        }
        Ok(Self {
            rows,
            cols,
            row_limit,
            col_limit,
        })
    }

    fn block(remaining: usize, limit: usize) -> usize {
        let count = remaining.min(limit);
        if remaining - count == 1 {
            count - 1
        } else {
            count
        }
    }
}

/// Shares the original joined owner, class, deadline, work account and buffer
/// implementation. No independent allowance or external cache is created.
pub(super) struct OutputTiledLeasedAwStorage<'a> {
    inner: LeasedAwStorage<'a>,
}

impl<'a> OutputTiledLeasedAwStorage<'a> {
    pub(super) fn new(
        owner: &'a Arc<SoaStorageBudget>,
        requested: Option<Instant>,
    ) -> Result<Self> {
        Ok(Self {
            inner: LeasedAwStorage::new(owner, requested)?,
        })
    }

    fn product_observed(
        &mut self,
        lhs: MatRef<'_, f64>,
        rhs: MatRef<'_, f64>,
        mut after_tile: impl FnMut(usize) -> Result<()>,
    ) -> Result<BoundedMat<f64>> {
        if lhs.ncols() != rhs.nrows() {
            return Err(invalid("tiled AW contraction mismatch"));
        }
        let k = lhs.ncols();
        let plan = OutputPlan::new(lhs.nrows(), k, rhs.ncols())?;
        self.checkpoint()?;
        let mut dst = self.matrix_zeros(plan.rows, plan.cols)?;
        let mut completed = 0;
        let mut row = 0;
        while row < plan.rows {
            let rows = OutputPlan::block(plan.rows - row, plan.row_limit);
            let mut col = 0;
            while col < plan.cols {
                let cols = OutputPlan::block(plan.cols - col, plan.col_limit);
                // Reuse the original audited admission and pre-work debit.
                // It also checks the original absolute authority immediately
                // before each indivisible native call, including last tiles.
                self.inner.before_product(rows, k, cols)?;
                crate::faer_parallelism::mat_mul_f64_views_into(
                    dst.view_mut().submatrix_mut(row, col, rows, cols),
                    lhs.submatrix(row, 0, rows, k),
                    rhs.submatrix(0, col, k, cols),
                );
                completed += 1;
                after_tile(completed)?;
                self.checkpoint()?;
                col += cols;
            }
            row += rows;
        }
        self.checkpoint()?;
        Ok(dst)
    }
}

impl AwStorage for OutputTiledLeasedAwStorage<'_> {
    type Matrix = BoundedMat<f64>;
    type Output = BoundedArray<f64>;

    fn matrix_ref(matrix: &Self::Matrix) -> MatRef<'_, f64> {
        LeasedAwStorage::matrix_ref(matrix)
    }
    fn matrix_mut(matrix: &mut Self::Matrix) -> MatMut<'_, f64> {
        LeasedAwStorage::matrix_mut(matrix)
    }
    fn output_slice(output: &mut Self::Output) -> &mut [f64] {
        LeasedAwStorage::output_slice(output)
    }
    fn matrix_from_fn(
        &mut self,
        rows: usize,
        cols: usize,
        getter: impl FnMut(usize, usize) -> f64,
    ) -> Result<Self::Matrix> {
        self.inner.matrix_from_fn(rows, cols, getter)
    }
    fn matrix_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Matrix> {
        self.inner.matrix_zeros(rows, cols)
    }
    fn output_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Output> {
        self.inner.output_zeros(rows, cols)
    }
    fn admit_products(
        &mut self,
        m: usize,
        k: usize,
        p: usize,
        rows_per_chunk: usize,
        cached: bool,
    ) -> Result<()> {
        self.checkpoint()?;
        if cached {
            return Err(NyError::UnsupportedOp(
                "bounded tiled AW: external widened cache has no retained owner binding".into(),
            ));
        }
        if rows_per_chunk == 0 {
            return Err(invalid("AW row chunk must be nonzero"));
        }
        LeasedAwStorage::product_units(m, k, p)?;
        OutputPlan::new(m.min(rows_per_chunk), k, p)?;
        let tail = m % rows_per_chunk;
        if tail != 0 {
            OutputPlan::new(tail, k, p)?;
        }
        Ok(())
    }
    fn before_copy_pair(&mut self, rows: usize, cols: usize) -> Result<()> {
        self.inner.before_copy_pair(rows, cols)
    }
    fn checkpoint(&mut self) -> Result<()> {
        self.inner.checkpoint()
    }
    fn product(&mut self, lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> Result<Self::Matrix> {
        self.product_observed(lhs, rhs, |_| Ok(()))
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[path = "output_tiled_tests.rs"]
mod tests;

#[path = "bounded_cpu.rs"]
mod bounded_cpu;
