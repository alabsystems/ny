// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ONE historical signed-AW/absolute-sum scalar reduction. The ordinary
//! fallback retains its exact polling cadence and owning allocation. The new
//! cooperative entry uses PRIVATE caller-owned views and original authority.
//! Neither entry partitions k, adds intermediate sums or allocates payload.

use super::{
    check_optional_aw_deadline, deadline_aw_faer_rows_per_chunk, f32_to_f64_exact,
    DEADLINE_AW_FAER_CHUNK_MACS,
};
use crate::amortise::{CooperativeWork, POLL_CHUNK_UNITS};
use faer::MatRef;
use ndarray::{ArrayView2, ArrayViewMut2};
use ny_core::{NyError, Result};
use std::time::Instant;

trait ScalarPoll {
    fn row(&mut self) -> Result<()>;
    fn before_k(&mut self) -> Result<()> {
        Ok(())
    }
    fn skipped_k(&mut self) -> Result<()>;
    fn columns(&self) -> usize;
    fn before_columns(&mut self, _: usize) -> Result<()> {
        Ok(())
    }
    fn after_columns(&mut self) -> Result<()>;
}

struct Legacy {
    deadline: Option<Instant>,
    columns: usize,
    skipped_quantum: usize,
    skipped_k_since_poll: usize,
}
impl ScalarPoll for Legacy {
    fn row(&mut self) -> Result<()> {
        check_optional_aw_deadline(self.deadline, "during certified f64 A·W")
    }
    fn skipped_k(&mut self) -> Result<()> {
        self.skipped_k_since_poll += 1;
        if self.skipped_k_since_poll >= self.skipped_quantum {
            check_optional_aw_deadline(self.deadline, "during certified f64 A·W")?;
            self.skipped_k_since_poll = 0;
        }
        Ok(())
    }
    fn columns(&self) -> usize {
        self.columns
    }
    fn after_columns(&mut self) -> Result<()> {
        check_optional_aw_deadline(self.deadline, "during certified f64 A·W")
    }
}

const ENTRY_UNITS: usize = 64;
const PAIR_UNITS: usize = 64;
const K_UNITS: usize = 64;
const MAC_UNITS: usize = 64;

struct Cooperative<'a, P>(&'a mut P);
impl<P: CooperativeWork> ScalarPoll for Cooperative<'_, P> {
    fn row(&mut self) -> Result<()> {
        self.0.checkpoint()
    }
    fn before_k(&mut self) -> Result<()> {
        self.0.before(K_UNITS)
    }
    fn skipped_k(&mut self) -> Result<()> {
        Ok(())
    }
    fn columns(&self) -> usize {
        POLL_CHUNK_UNITS / MAC_UNITS
    }
    fn before_columns(&mut self, count: usize) -> Result<()> {
        self.0.before(count * MAC_UNITS)
    }
    fn after_columns(&mut self) -> Result<()> {
        self.0.checkpoint()
    }
}

// Only this function contains signed or absolute-sum dot-product arithmetic.
// Caller storage/getters must be fixed-shape, cheap scalar accesses with live
// original source claims. Output is private; discard BOTH planes on any error.
fn fill<P: ScalarPoll>(
    rows: usize,
    contraction: usize,
    columns: usize,
    mut a_value: impl FnMut(usize, usize) -> f32,
    w: MatRef<'_, f32>,
    mut a64: ArrayViewMut2<'_, f64>,
    mut s: ArrayViewMut2<'_, f64>,
    poll: &mut P,
) -> Result<()> {
    let m = rows;
    let k = contraction;
    let p = columns;
    for i in 0..m {
        poll.row()?;
        let mut a64_row = a64.row_mut(i);
        let mut s_row = s.row_mut(i);
        for kk in 0..k {
            poll.before_k()?;
            let av = f32_to_f64_exact(a_value(i, kk));
            if av == 0.0 || p == 0 {
                poll.skipped_k()?;
                continue;
            }

            let av_abs = av.abs();
            let mut j_start = 0usize;
            while j_start < p {
                let j_end = (j_start + poll.columns()).min(p);
                poll.before_columns(j_end - j_start)?;
                for j in j_start..j_end {
                    let wv = f32_to_f64_exact(w[(kk, j)]);
                    a64_row[j] += av * wv;
                    s_row[j] += av_abs * wv.abs();
                }
                poll.after_columns()?;
                j_start = j_end;
            }
        }
    }
    Ok(())
}

/// Existing allocating fallback's bridge. Its caller has validated source
/// dimensions and allocated/checked these original contiguous zeroed planes.
/// The two quanta are the unchanged constants from that original wrapper.
pub(super) fn legacy_into(
    a: MatRef<'_, f32>,
    w: MatRef<'_, f32>,
    a64: &mut [f64],
    s: &mut [f64],
    deadline: Option<Instant>,
    columns: usize,
    skipped_quantum: usize,
) -> Result<()> {
    assert!(columns > 0 && skipped_quantum > 0);
    let shape = (a.nrows(), w.ncols());
    let a64 = ArrayViewMut2::from_shape(shape, a64).expect("original AW output shape");
    let s = ArrayViewMut2::from_shape(shape, s).expect("original AW abs-sum shape");
    let mut poll = Legacy {
        deadline,
        columns,
        skipped_quantum,
        skipped_k_since_poll: 0,
    };
    fill(
        a.nrows(),
        a.ncols(),
        w.ncols(),
        |i, k| a[(i, k)],
        w,
        a64,
        s,
        &mut poll,
    )?;
    check_optional_aw_deadline(deadline, "during certified f64 A·W")
}

fn validate(
    rows: usize,
    contraction: usize,
    w: MatRef<'_, f32>,
    a64: &ArrayViewMut2<'_, f64>,
    s: &ArrayViewMut2<'_, f64>,
) -> Result<()> {
    let shape = (rows, w.ncols());
    if w.nrows() != contraction || a64.dim() != shape || s.dim() != shape {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, w.ncols(), contraction],
            got: vec![a64.nrows(), a64.ncols(), s.nrows(), s.ncols(), w.nrows()],
        });
    }
    Ok(())
}

fn cooperative<P: CooperativeWork>(
    rows: usize,
    contraction: usize,
    a_value: impl FnMut(usize, usize) -> f32,
    w: MatRef<'_, f32>,
    mut a64: ArrayViewMut2<'_, f64>,
    mut s: ArrayViewMut2<'_, f64>,
    work: &mut P,
) -> Result<()> {
    // Shape is checked before ANY output write. Diagnostics can allocate;
    // this successful-payload sink is not an all-path allocator census.
    validate(rows, contraction, w, &a64, &s)?;
    if rows == 0 || w.ncols() == 0 {
        return work.checkpoint();
    }
    for i in 0..rows {
        for j in 0..w.ncols() {
            work.before(PAIR_UNITS)?;
            a64[[i, j]] = 0.0;
            s[[i, j]] = 0.0;
        }
    }
    fill(
        rows,
        contraction,
        w.ncols(),
        a_value,
        w,
        a64,
        s,
        &mut Cooperative(&mut *work),
    )?;
    work.checkpoint()
}

/// Exact original i/kk/j reduction with before-work debits under the caller's
/// original authority. Supports arbitrary strides, singleton/zero axes and
/// full-k contractions beyond native tile admission. No k partition or cache.
#[cfg(test)]
pub(crate) fn into_with_work<P: CooperativeWork>(
    a: MatRef<'_, f32>,
    w: MatRef<'_, f32>,
    a64: ArrayViewMut2<'_, f64>,
    s: ArrayViewMut2<'_, f64>,
    work: &mut P,
) -> Result<()> {
    work.checkpoint()?;
    work.before(ENTRY_UNITS)?;
    cooperative(a.nrows(), a.ncols(), |i, k| a[(i, k)], w, a64, s, work)
}

/// Borrow lower/upper position blocks directly; no stacked f32 input buffer
/// or hidden widening/absolute-weight cache is allocated by this adapter.
pub(crate) fn stacked_into_with_work<P: CooperativeWork>(
    lower: ArrayView2<'_, f32>,
    upper: ArrayView2<'_, f32>,
    offset: usize,
    contraction: usize,
    w: MatRef<'_, f32>,
    a64: ArrayViewMut2<'_, f64>,
    s: ArrayViewMut2<'_, f64>,
    work: &mut P,
) -> Result<()> {
    work.checkpoint()?;
    work.before(ENTRY_UNITS)?;
    let rows = lower.nrows();
    let total_rows = rows
        .checked_mul(2)
        .ok_or_else(|| NyError::InvalidSpec("stacked scalar AW rows overflow".into()))?;
    let end = offset
        .checked_add(contraction)
        .ok_or_else(|| NyError::InvalidSpec("stacked scalar AW columns overflow".into()))?;
    if lower.dim() != upper.dim() || end > lower.ncols() {
        return Err(NyError::ShapeMismatch {
            expected: vec![rows, end],
            got: vec![upper.nrows(), upper.ncols(), lower.ncols()],
        });
    }
    cooperative(
        total_rows,
        contraction,
        |i, k| {
            if i < rows {
                lower[[i, offset + k]]
            } else {
                upper[[i - rows, offset + k]]
            }
        },
        w,
        a64,
        s,
        work,
    )
}

/// Original native AW row plan and its exact historical MAC allowance.
/// This only exposes existing policy metadata; it creates no new allowance.
pub(crate) fn original_native_chunk_plan(k: usize, p: usize) -> Option<(usize, usize)> {
    deadline_aw_faer_rows_per_chunk(k, p, DEADLINE_AW_FAER_CHUNK_MACS)
        .map(|rows| (DEADLINE_AW_FAER_CHUNK_MACS, rows))
}

#[cfg(test)]
#[path = "scalar_aw_tests.rs"]
mod tests;
