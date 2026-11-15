// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One historical outward-stepped reduction, with either its original deadline
//! cadence or caller-supplied cooperative work and PRIVATE output storage.
//! The cooperative entry reads original signed weights directly, avoiding the
//! lazy absolute-weight cache and all GEMM/packing/scratch allocation.

use super::{next_up_nonnegative_f64, nonnegative_f32_error_or_infinity, publish_error_up_normal};
use crate::amortise::CooperativeWork;
use faer::MatRef;
use ndarray::{ArrayView2, ArrayViewMut2};
use ny_core::{NyError, Result};

pub(super) fn shape(
    error: ArrayView2<'_, f32>,
    column_offset: usize,
    contraction: usize,
    weights: MatRef<'_, f32>,
) -> Result<(usize, usize)> {
    let rows = error.nrows();
    let output_columns = weights.ncols();
    if error.ncols() < column_offset.saturating_add(contraction) || weights.nrows() != contraction {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, column_offset.saturating_add(contraction)],
            got: vec![error.nrows(), error.ncols()],
        });
    }
    Ok((rows, output_columns))
}

trait StepPoll {
    fn entry(&mut self) -> Result<()> {
        Ok(())
    }
    fn mac(&mut self) -> Result<()>;
    fn publication(&mut self) -> Result<()> {
        Ok(())
    }
}

struct LegacyPoll<'a, P> {
    check: &'a P,
    operations: usize,
    quantum: usize,
}

impl<P: Fn() -> Result<()>> StepPoll for LegacyPoll<'_, P> {
    fn mac(&mut self) -> Result<()> {
        if self.operations.is_multiple_of(self.quantum) {
            (self.check)()?;
        }
        self.operations = self.operations.wrapping_add(1);
        Ok(())
    }
}

struct Cooperative<'a, P>(&'a mut P);
impl<P: CooperativeWork> StepPoll for Cooperative<'_, P> {
    fn entry(&mut self) -> Result<()> {
        self.0.before(64)
    }
    fn mac(&mut self) -> Result<()> {
        self.0.before(64)
    }
    fn publication(&mut self) -> Result<()> {
        self.0.before(64)
    }
}

// The only scalar reduction body. Operand access/storage and polling vary;
// sanitization, zero-dominates-infinity, k order, stepped successor and final
// directed/normal publication are shared with the original production fallback.
fn fill<P: StepPoll>(
    error: ArrayView2<'_, f32>,
    column_offset: usize,
    contraction: usize,
    absolute_weight: impl Fn(usize, usize) -> f32,
    mut product: ArrayViewMut2<'_, f32>,
    poll: &mut P,
) -> Result<()> {
    let rows = product.nrows();
    let output_columns = product.ncols();
    for i in 0..rows {
        for j in 0..output_columns {
            poll.entry()?;
            let mut stepped = 0.0f64;
            for kk in 0..contraction {
                poll.mac()?;
                let error_value = nonnegative_f32_error_or_infinity(error[[i, column_offset + kk]]);
                let weight_abs = nonnegative_f32_error_or_infinity(absolute_weight(kk, j));
                let term = if error_value == 0.0 || weight_abs == 0.0 {
                    0.0
                } else {
                    error_value * weight_abs
                };
                // Certified coefficient errors and |W| are non-negative. A
                // negative or NaN term violates that invariant; +inf is the
                // conservative error enclosure. The stepped arm takes one
                // successor after every RN addition, which dominates its exact
                // real sum without any contraction-width assumption.
                if term == 0.0 {
                    continue;
                }
                if term > 0.0 {
                    stepped = next_up_nonnegative_f64(stepped + term);
                } else {
                    stepped = f64::INFINITY;
                }
            }
            poll.publication()?;
            product[[i, j]] = publish_error_up_normal(stepped);
        }
    }
    Ok(())
}

/// Original scalar fallback's already-absolute operand and exact MAC cadence.
pub(super) fn legacy_into(
    error: ArrayView2<'_, f32>,
    column_offset: usize,
    contraction: usize,
    weights_abs: MatRef<'_, f32>,
    product: ArrayViewMut2<'_, f32>,
    quantum: usize,
    check: &impl Fn() -> Result<()>,
) -> Result<()> {
    fill(
        error,
        column_offset,
        contraction,
        |i, j| weights_abs[(i, j)],
        product,
        &mut LegacyPoll {
            check,
            operations: 0,
            quantum: quantum.max(1),
        },
    )
}

/// Fill a PRIVATE caller-owned output under the caller's original authority.
/// Source captures and the destination's actual capacity must remain claimed
/// by the caller throughout. An error requires discarding the private output.
/// No cache, allocation, GEMM, new budget/lease or owned output is created.
///
/// `weights` is the original SIGNED weight, not an absolute-cache receipt.
/// Clearing its sign bit reproduces absolute-cache values before the SAME
/// nonnegative sanitizer, including true f32 subnormals and NaN/Inf poison.
/// Entry/init and publication each cost64, every full-k MAC costs64; even a
/// zero contraction/output gets authoritative entry and final checkpoints.
pub(crate) fn into_with_work<P: CooperativeWork>(
    error: ArrayView2<'_, f32>,
    column_offset: usize,
    contraction: usize,
    weights: MatRef<'_, f32>,
    product: ArrayViewMut2<'_, f32>,
    work: &mut P,
) -> Result<()> {
    work.checkpoint()?;
    work.before(64)?;
    let (rows, cols) = shape(error, column_offset, contraction, weights)?;
    if product.dim() != (rows, cols) {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, cols],
            got: vec![product.nrows(), product.ncols()],
        });
    }
    fill(
        error,
        column_offset,
        contraction,
        |i, j| f32::from_bits(weights[(i, j)].to_bits() & 0x7fff_ffff),
        product,
        &mut Cooperative(&mut *work),
    )?;
    work.checkpoint()
}

#[cfg(test)]
#[path = "incoming_error_step_tests.rs"]
mod tests;
