// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Leased storage for the shared CPU kernel's future allocation adapter.
//!
//! Source captures must already be covered by retained bytes or another live
//! lease. A getter copies/converts one scalar in bounded work; this builder
//! neither certifies its provenance nor performs any bound arithmetic.
//! The complete CPU kernel, its cache, GEMM scratch and output adoption still
//! need integration before finite collector admission can change.

use super::{element_units, invalid, narrower, reserve, Charge, Poll, SoaStorageBudget, ENTRY};
use faer::{MatMut, MatRef};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use ny_core::Result;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

const AFTER_RESERVE: &str = "bounded matrix after reserve";
const FILL: &str = "bounded matrix fill";
const COMMIT: &str = "bounded matrix commit";

mod scalar {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

#[repr(C, align(64))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct F32Block([f32; 16]);
#[repr(C, align(64))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct F64Block([f64; 8]);
const _: () = assert!(size_of::<F32Block>() == 64 && size_of::<F64Block>() == 64);
const _: () = assert!(align_of::<F32Block>() == 64 && align_of::<F64Block>() == 64);

/// Sealed scalar/block pairs make safe slice casts retain every bit.
pub(super) trait Scalar: scalar::Sealed + Copy + Default + bytemuck::Pod {
    type Block: Copy + bytemuck::Pod;
    fn block(getter: impl FnMut(usize) -> Self) -> Self::Block;
}
impl Scalar for f32 {
    type Block = F32Block;
    fn block(getter: impl FnMut(usize) -> Self) -> Self::Block {
        F32Block(std::array::from_fn(getter))
    }
}
impl Scalar for f64 {
    type Block = F64Block;
    fn block(getter: impl FnMut(usize) -> Self) -> Self::Block {
        F64Block(std::array::from_fn(getter))
    }
}

struct BufferPoll {
    shared: Poll,
    #[cfg(test)]
    fail_after_units: Option<usize>,
    #[cfg(test)]
    observed_units: usize,
}

impl BufferPoll {
    fn new(deadline: Option<Instant>) -> Self {
        Self {
            shared: Poll::new(deadline),
            #[cfg(test)]
            fail_after_units: None,
            #[cfg(test)]
            observed_units: 0,
        }
    }

    fn for_owner(owner: &Arc<SoaStorageBudget>, requested: Option<Instant>) -> Self {
        let mut result = Self::new(narrower(owner.deadline, requested));
        result.shared = Poll::for_owner(owner, requested);
        result
    }

    fn work<T>(&mut self) -> Result<()> {
        let units = element_units::<T>();
        #[cfg(test)]
        {
            self.observed_units = self.observed_units.saturating_add(units);
            if self
                .fail_after_units
                .is_some_and(|after| self.observed_units >= after)
            {
                return Err(ny_core::NyError::DeadlineExceeded(format!(
                    "bounded matrix forced expiry after {} actual work units",
                    self.observed_units
                )));
            }
        }
        self.shared.work(units, FILL)
    }
}

fn cells(rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(invalid("matrix dimensions must be nonzero"));
    }
    rows.checked_mul(cols)
        .ok_or_else(|| invalid("matrix shape overflow"))
}

fn bytes<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| invalid("matrix byte count overflow"))
}

/// faer 0.24.0 rounds plain float row capacities to a 64-byte boundary.
/// Admission uses that padded capacity, never just the logical element count.
/// The aligned block Vec reconciles its actual capacity BEFORE any fill.
fn padded_cells<T: Scalar>(rows: usize, cols: usize) -> Result<usize> {
    cells(rows, cols)?;
    let alignment = 64 / size_of::<T>();
    let rows = rows
        .checked_next_multiple_of(alignment)
        .ok_or_else(|| invalid("faer row capacity overflow"))?;
    rows.checked_mul(cols)
        .ok_or_else(|| invalid("faer padded shape overflow"))
}

/// Storage drops before its charge; only fixed-shape views permit mutation.
pub(super) struct BoundedArray<T> {
    storage: Array2<T>,
    _charge: Charge,
    deadline: Option<Instant>,
}

impl<T: Scalar> BoundedArray<T> {
    fn with_poll(
        owner: &Arc<SoaStorageBudget>,
        rows: usize,
        cols: usize,
        deadline: Option<Instant>,
        getter: impl FnMut(usize, usize) -> T,
        poll: &mut BufferPoll,
    ) -> Result<Self> {
        Self::with_poll_layout(owner, rows, cols, deadline, false, getter, poll)
    }

    fn with_poll_layout(
        owner: &Arc<SoaStorageBudget>,
        rows: usize,
        cols: usize,
        deadline: Option<Instant>,
        allow_empty: bool,
        mut getter: impl FnMut(usize, usize) -> T,
        poll: &mut BufferPoll,
    ) -> Result<Self> {
        poll.shared.checkpoint(ENTRY)?;
        let length = if allow_empty {
            rows.checked_mul(cols)
                .ok_or_else(|| invalid("matrix shape overflow"))?
        } else {
            cells(rows, cols)?
        };
        let mut charge = owner.charge(bytes::<T>(length)?, &mut poll.shared)?;
        let mut values = reserve::<T>(length, &mut charge, &mut poll.shared)?;
        poll.shared.checkpoint(AFTER_RESERVE)?;
        poll.shared
            .admit_elements(length, element_units::<T>(), FILL)?;
        // An empty column axis must not spin over a huge row dimension.
        if length != 0 {
            for row in 0..rows {
                for col in 0..cols {
                    values.push(getter(row, col));
                    poll.work::<T>()?;
                }
            }
        }
        let storage = Array2::from_shape_vec((rows, cols), values)
            .map_err(|_| invalid("matrix staged shape mismatch"))?;
        owner.check_live(&mut poll.shared)?;
        charge.materialize()?;
        poll.shared.checkpoint(COMMIT)?;
        Ok(Self {
            storage,
            _charge: charge,
            deadline,
        })
    }

    pub(super) fn try_from_fn(
        owner: &Arc<SoaStorageBudget>,
        rows: usize,
        cols: usize,
        requested: Option<Instant>,
        getter: impl FnMut(usize, usize) -> T,
    ) -> Result<Self> {
        let deadline = narrower(owner.deadline, requested);
        Self::with_poll(
            owner,
            rows,
            cols,
            deadline,
            getter,
            &mut BufferPoll::for_owner(owner, deadline),
        )
    }
    /// Explicit empty-axis storage. The historical constructor still refuses
    /// zero dimensions. This variant keeps the SAME reserve/capacity/claim/fill
    /// pipeline and entry/final authority, including for zero payload bytes.
    pub(super) fn try_from_fn_allow_empty(
        owner: &Arc<SoaStorageBudget>,
        rows: usize,
        cols: usize,
        requested: Option<Instant>,
        getter: impl FnMut(usize, usize) -> T,
    ) -> Result<Self> {
        let deadline = narrower(owner.deadline, requested);
        Self::with_poll_layout(
            owner,
            rows,
            cols,
            deadline,
            true,
            getter,
            &mut BufferPoll::for_owner(owner, deadline),
        )
    }
    pub(super) fn array(&self) -> &Array2<T> {
        &self.storage
    }
    pub(super) fn view(&self) -> ArrayView2<'_, T> {
        self.storage.view()
    }
    pub(super) fn view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.storage.view_mut()
    }
    pub(super) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

/// Leased 64-byte-aligned storage with faer-compatible column-major views.
/// Safe views retain the original row padding and permit no capacity growth.
pub(super) struct BoundedMat<T: Scalar> {
    storage: Vec<T::Block>,
    _charge: Charge,
    rows: usize,
    cols: usize,
    stride: usize,
    deadline: Option<Instant>,
}

impl<T: Scalar> BoundedMat<T> {
    fn with_poll(
        owner: &Arc<SoaStorageBudget>,
        rows: usize,
        cols: usize,
        deadline: Option<Instant>,
        mut getter: impl FnMut(usize, usize) -> T,
        poll: &mut BufferPoll,
    ) -> Result<Self> {
        poll.shared.checkpoint(ENTRY)?;
        let capacity = padded_cells::<T>(rows, cols)?;
        let payload = bytes::<T>(capacity)?;
        let per_block = size_of::<T::Block>() / size_of::<T>();
        let stride = capacity / cols;
        if per_block == 0 || !stride.is_multiple_of(per_block) {
            return Err(invalid("aligned matrix block layout mismatch"));
        }
        let blocks = capacity / per_block;
        let mut charge = owner.charge(payload, &mut poll.shared)?;
        let mut storage = reserve::<T::Block>(blocks, &mut charge, &mut poll.shared)?;
        poll.shared.checkpoint(AFTER_RESERVE)?;
        poll.shared
            .admit_elements(blocks, element_units::<T::Block>(), FILL)?;
        for col in 0..cols {
            for start in (0..stride).step_by(per_block) {
                let block = T::block(|offset| {
                    let row = start + offset;
                    if row < rows {
                        getter(row, col)
                    } else {
                        T::default()
                    }
                });
                storage.push(block);
                poll.work::<T::Block>()?;
            }
        }
        owner.check_live(&mut poll.shared)?;
        charge.materialize()?;
        poll.shared.checkpoint(COMMIT)?;
        Ok(Self {
            storage,
            _charge: charge,
            rows,
            cols,
            stride,
            deadline,
        })
    }
    pub(super) fn try_from_fn(
        owner: &Arc<SoaStorageBudget>,
        rows: usize,
        cols: usize,
        requested: Option<Instant>,
        getter: impl FnMut(usize, usize) -> T,
    ) -> Result<Self> {
        let deadline = narrower(owner.deadline, requested);
        Self::with_poll(
            owner,
            rows,
            cols,
            deadline,
            getter,
            &mut BufferPoll::for_owner(owner, deadline),
        )
    }
    pub(super) fn matrix(&self) -> MatRef<'_, T> {
        MatRef::from_column_major_slice_with_stride(
            bytemuck::cast_slice(&self.storage),
            self.rows,
            self.cols,
            self.stride,
        )
    }
    pub(super) fn view(&self) -> MatRef<'_, T> {
        self.matrix()
    }
    pub(super) fn view_mut(&mut self) -> MatMut<'_, T> {
        MatMut::from_column_major_slice_with_stride_mut(
            bytemuck::cast_slice_mut(&mut self.storage),
            self.rows,
            self.cols,
            self.stride,
        )
    }
    pub(super) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

#[cfg(test)]
#[path = "bounded_matrix_tests.rs"]
mod tests;

#[path = "bounded_aw.rs"]
mod bounded_aw;

#[path = "claimed_output.rs"]
mod claimed_output;
