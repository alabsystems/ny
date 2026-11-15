// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Existing-query storage for the shared AW body. Only the audited scratch-free
//! native f64 dispatch is accepted. This is not a complete finite CPU kernel.

use super::*;
use crate::amortise::POLL_CHUNK_UNITS;
use crate::layers::linear::crown_single::aw_storage::AwStorage;
use ny_core::NyError;

/// The caller must retain the original f32 inputs under this owner's existing
/// producer claim. The full graph/maps/seed correspondence is still a caller
/// obligation; accepting buffers here never certifies their provenance.
pub(super) struct LeasedAwStorage<'a> {
    owner: &'a Arc<SoaStorageBudget>,
    deadline: Option<Instant>,
    poll: Poll,
}

impl<'a> LeasedAwStorage<'a> {
    pub(super) fn new(
        owner: &'a Arc<SoaStorageBudget>,
        requested: Option<Instant>,
    ) -> Result<Self> {
        if owner.query.is_none() {
            return Err(invalid("AW storage requires the original joined query"));
        }
        let deadline = narrower(owner.deadline, requested);
        let mut result = Self {
            owner,
            deadline,
            poll: Poll::for_owner(owner, deadline),
        };
        result.checkpoint()?;
        Ok(result)
    }

    fn product_units(rows: usize, k: usize, cols: usize) -> Result<usize> {
        // Four f32-equivalent primitive units per f64 multiply plus add.
        rows.checked_mul(k)
            .and_then(|n| n.checked_mul(cols))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("AW product work overflow"))
    }

    fn admit_shape(rows: usize, k: usize, cols: usize) -> Result<()> {
        // faer 0.24.0's audited aarch64 native f64 path reaches planless
        // nano-gemm at <= 16^3 MACs, with no packing slab or heap workspace.
        // Exclude the earlier matvec/rank-update dispatches, including a
        // singleton final row block. The original k contraction stays whole.
        // This stricter work quantum (<= 1024 MACs) is inside that threshold.
        if !cfg!(target_arch = "aarch64")
            || rows <= 1
            || k <= 1
            || cols <= 1
            || Self::product_units(rows, k, cols)? > POLL_CHUNK_UNITS
        {
            return Err(NyError::UnsupportedOp(
                "bounded AW: GEMM scratch or cooperative work is not accounted for".into(),
            ));
        }
        Ok(())
    }
}

impl AwStorage for LeasedAwStorage<'_> {
    type Matrix = BoundedMat<f64>;
    type Output = BoundedArray<f64>;

    fn matrix_ref(matrix: &Self::Matrix) -> MatRef<'_, f64> {
        matrix.view()
    }
    fn matrix_mut(matrix: &mut Self::Matrix) -> MatMut<'_, f64> {
        matrix.view_mut()
    }
    fn output_slice(output: &mut Self::Output) -> &mut [f64] {
        output
            .storage
            .as_slice_mut()
            .expect("fixed row-major output")
    }
    fn matrix_from_fn(
        &mut self,
        rows: usize,
        cols: usize,
        getter: impl FnMut(usize, usize) -> f64,
    ) -> Result<Self::Matrix> {
        BoundedMat::try_from_fn(self.owner, rows, cols, self.deadline, getter)
    }
    fn matrix_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Matrix> {
        self.matrix_from_fn(rows, cols, |_, _| 0.0)
    }
    fn output_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Output> {
        BoundedArray::try_from_fn(self.owner, rows, cols, self.deadline, |_, _| 0.0)
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
                "bounded AW: external widened cache has no retained owner binding".into(),
            ));
        }
        if rows_per_chunk == 0 {
            return Err(invalid("AW row chunk must be nonzero"));
        }
        Self::admit_shape(m.min(rows_per_chunk), k, p)?;
        let tail = m % rows_per_chunk;
        if tail != 0 {
            Self::admit_shape(tail, k, p)?;
        }
        Ok(())
    }
    fn before_product(&mut self, rows: usize, k: usize, cols: usize) -> Result<()> {
        Self::admit_shape(rows, k, cols)?;
        self.checkpoint()?;
        self.poll
            .admit_elements(Self::product_units(rows, k, cols)?, 1, FILL)
    }
    fn before_copy_pair(&mut self, rows: usize, cols: usize) -> Result<()> {
        self.checkpoint()?;
        self.poll.admit_elements(cells(rows, cols)?, 4, FILL)
    }
    fn checkpoint(&mut self) -> Result<()> {
        self.owner.check_live(&mut self.poll)
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[path = "bounded_aw_tests.rs"]
mod tests;

// Separate arithmetic policy: output tiles may select a different faer kernel
// and reduction order. The original full-width storage policy stays unchanged.
#[path = "bounded_aw/output_tiled.rs"]
mod output_tiled;
