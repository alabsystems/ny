// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Paid ordinary y-row passes and an immutable complete root memo owner.
//!
//! The actual ordinary backward kernel runs against the immutable projection
//! of the genuine completed root. Every output column is an original Gemm1
//! identity row. All four coefficient/error planes, independent constants,
//! concretized endpoints and owner metadata are admitted before allocation.
//! Partial tiles never publish a memo. Clones share the original source and
//! one last-owner reservation; neither raw PassOut nor mutable RootGates can
//! escape. RootEval/YPack must retain this owner when borrowing these views.
//! Their ordinary caller ingress is a separate integration requirement.

use super::ir::{checkpoint, invalid, refusal};
use super::root_adapter::PreparedRootAdapter;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use crate::margin_row::engine::{error_mul, run_prepared_ordinary_tile, LaneDir, PassOut, Seed};
use crate::margin_row::net::{nominal_product_dust, TwinOp};
use crate::margin_row::rounding::{certify_up, gamma_n, next_down, next_up};
use ndarray::{Array2, ArrayView2};
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

const MAX_TILE: usize = 64;
const MAX_OPS: usize = 4096;
const MAX_CELLS: usize = 16 * 1024 * 1024;
const META: usize = 4096;
const POLL: usize = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::margin_row) struct PreparedOrdinaryPassReceipt {
    pub lower_columns: usize,
    pub upper_columns: usize,
    pub tiles: usize,
    pub backward_passes: usize,
    pub max_tile_columns: usize,
    pub tile_admission_attempts: usize,
    pub stored_coefficient_cells: usize,
    pub stored_error_cells: usize,
    pub concretized_columns: usize,
}

struct Planes {
    a: Array2<f64>,
    e: Array2<f64>,
    b: Vec<f64>,
    eb: Vec<f64>,
}
struct OrdinaryInner {
    adapter: PreparedRootAdapter,
    lower: Planes,
    upper: Planes,
    lo: Vec<f64>,
    hi: Vec<f64>,
    deadline: Instant,
    receipt: PreparedOrdinaryPassReceipt,
    resident: PersistentReservation,
}

/// Immutable y-row evidence and complete numerical memo, never a verdict.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedOrdinaryRows(Arc<OrdinaryInner>, Instant);

pub(in crate::margin_row) struct PreparedOrdinaryMovedRows {
    lower: PassOut,
    upper: PassOut,
    lo: Vec<f64>,
    hi: Vec<f64>,
    owner: PreparedOrdinaryRowOwner,
}

/// The original payload claim moves exactly once with its actual arrays. Its
/// constructor is private; neither copied row facts nor a supplied claim can
/// create this transport owner.
pub(in crate::margin_row) struct PreparedOrdinaryRowOwner {
    adapter: PreparedRootAdapter,
    deadline: Instant,
    receipt: PreparedOrdinaryPassReceipt,
    resident: PersistentReservation,
}

impl PreparedOrdinaryRowOwner {
    pub(in crate::margin_row) fn adapter(&self) -> &PreparedRootAdapter {
        &self.adapter
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.adapter.deadline(Some(
            requested.map_or(self.deadline, |at| at.min(self.deadline)),
        ))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.adapter.checkpoint(Some(at))?;
        let net = self.adapter.net();
        if !self.resident.shares_ledger(self.adapter.source().budget())
            || self.resident_bytes() < owner_bytes(net.n_in, net.n_y)?
            || self.receipt.lower_columns != net.n_y
            || self.receipt.upper_columns != net.n_y
            || self.receipt.concretized_columns != mul(net.n_y, 2)?
        {
            return Err(invalid("ordinary moved rows lost original complete owner"));
        }
        Ok(())
    }
}

impl PreparedOrdinaryMovedRows {
    /// Numerical transport stays inside margin_row and always accompanies its
    /// sealed original owner. This does not mint an imported source or verdict.
    pub(in crate::margin_row) fn into_parts(
        self,
    ) -> (
        PassOut,
        PassOut,
        Vec<f64>,
        Vec<f64>,
        PreparedOrdinaryRowOwner,
    ) {
        (self.lower, self.upper, self.lo, self.hi, self.owner)
    }
}

impl PreparedOrdinaryRows {
    /// Dispatch the actual atomic ownership result without retaining both
    /// differently sized transport variants. Both consuming continuations are
    /// monomorphized; only the selected branch runs and neither allocates here.
    pub(in crate::margin_row) fn with_root_rows<T>(
        self,
        requested: Option<Instant>,
        unique: impl FnOnce(PreparedOrdinaryMovedRows) -> Result<T>,
        shared_action: impl FnOnce(Self) -> Result<T>,
    ) -> Result<T> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let inner = match Arc::try_unwrap(self.0) {
            Ok(inner) => inner,
            Err(shared) => return shared_action(Self(shared, at)),
        };
        let OrdinaryInner {
            adapter,
            lower,
            upper,
            lo,
            hi,
            deadline,
            receipt,
            resident,
        } = inner;
        let owner = PreparedOrdinaryRowOwner {
            adapter,
            deadline: at.min(deadline),
            receipt,
            resident,
        };
        let pass = |p: Planes| PassOut {
            a: p.a,
            e: Some(p.e),
            b: p.b,
            eb: p.eb,
            coll: None,
            coll_rows: None,
            unst_rows: None,
        };
        let moved = PreparedOrdinaryMovedRows {
            lower: pass(lower),
            upper: pass(upper),
            lo,
            hi,
            owner,
        };
        moved.owner.checkpoint(Some(at))?;
        unique(moved)
    }

    /// Explicit shared-owner fallback. The complete old/new payload overlap
    /// is admitted on the same ledger before any copy. This preserves cached
    /// or sibling consumers rather than forcing them to surrender ownership.
    pub(in crate::margin_row) fn copy_into_root_rows(
        self,
        requested: Option<Instant>,
    ) -> Result<PreparedOrdinaryMovedRows> {
        let mut at = self.deadline(requested)?;
        let (source_lo, source_hi) = self.box_bounds(Some(at))?;
        let budget = self.adapter().source().budget();
        let n = self.n_in();
        let columns = self.columns();
        let mut lease = admit(
            budget,
            WorkRequest {
                persistent_bytes: owner_bytes(n, columns)?,
                work: work(add(mul(mul(n, columns)?, 8)?, mul(columns, 16)?)?, 256)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        at = min_lease(at, &lease);
        let copy_plane = |lower: bool| -> Result<PassOut> {
            let (coefficients, errors, bias, bias_errors) = self.matrices(lower, Some(at))?;
            let cells = mul(n, columns)?;
            let copy = |src: &[f64]| -> Result<Vec<f64>> {
                let mut dst = filled(src.len(), budget, at)?;
                for (i, (&v, out)) in src.iter().zip(&mut dst).enumerate() {
                    poll(budget, at, i)?;
                    *out = v;
                }
                checkpoint(budget, Some(at))?;
                Ok(dst)
            };
            let a = coefficients
                .as_slice()
                .ok_or_else(|| invalid("ordinary copied row layout"))?;
            let e = errors
                .as_slice()
                .ok_or_else(|| invalid("ordinary copied error layout"))?;
            if a.len() != cells || e.len() != cells {
                return Err(invalid("ordinary copied row dimensions"));
            }
            Ok(PassOut {
                a: Array2::from_shape_vec((n, columns), copy(a)?)
                    .map_err(|_| invalid("ordinary copied coefficient shape"))?,
                e: Some(
                    Array2::from_shape_vec((n, columns), copy(e)?)
                        .map_err(|_| invalid("ordinary copied error shape"))?,
                ),
                b: copy(bias)?,
                eb: copy(bias_errors)?,
                coll: None,
                coll_rows: None,
                unst_rows: None,
            })
        };
        let lower = copy_plane(true)?;
        let upper = copy_plane(false)?;
        let copy_box = |src: &[f64]| -> Result<Vec<f64>> {
            let mut dst = filled(src.len(), budget, at)?;
            for (i, (&v, out)) in src.iter().zip(&mut dst).enumerate() {
                poll(budget, at, i)?;
                *out = v;
            }
            Ok(dst)
        };
        let lo = copy_box(source_lo)?;
        let hi = copy_box(source_hi)?;
        self.checkpoint(Some(at))?;
        lease.mark_materialized().map_err(refusal)?;
        let owner = PreparedOrdinaryRowOwner {
            adapter: self.adapter().clone(),
            deadline: at,
            receipt: self.receipt(),
            resident: lease.commit_persistent().map_err(refusal)?,
        };
        let moved = PreparedOrdinaryMovedRows {
            lower,
            upper,
            lo,
            hi,
            owner,
        };
        moved.owner.checkpoint(Some(at))?;
        Ok(moved)
    }
    #[cfg(test)]
    pub(in crate::margin_row) fn storage_pointers(&self) -> [usize; 10] {
        [
            self.0.lower.a.as_ptr() as usize,
            self.0.lower.e.as_ptr() as usize,
            self.0.lower.b.as_ptr() as usize,
            self.0.lower.eb.as_ptr() as usize,
            self.0.upper.a.as_ptr() as usize,
            self.0.upper.e.as_ptr() as usize,
            self.0.upper.b.as_ptr() as usize,
            self.0.upper.eb.as_ptr() as usize,
            self.0.lo.as_ptr() as usize,
            self.0.hi.as_ptr() as usize,
        ]
    }
    pub(in crate::margin_row) fn adapter(&self) -> &PreparedRootAdapter {
        &self.0.adapter
    }
    pub(in crate::margin_row) fn receipt(&self) -> PreparedOrdinaryPassReceipt {
        self.0.receipt
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn n_in(&self) -> usize {
        self.adapter().net().n_in
    }
    pub(in crate::margin_row) fn columns(&self) -> usize {
        self.adapter().net().n_y
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.adapter()
            .deadline(Some(requested.map_or(self.1.min(self.0.deadline), |at| {
                at.min(self.1).min(self.0.deadline)
            })))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.adapter().checkpoint(Some(at))?;
        if !self
            .0
            .resident
            .shares_ledger(self.adapter().source().budget())
            || self.resident_bytes() < owner_bytes(self.n_in(), self.columns())?
            || self.0.receipt.lower_columns != self.columns()
            || self.0.receipt.upper_columns != self.columns()
            || self.0.receipt.concretized_columns != mul(self.columns(), 2)?
        {
            return Err(invalid("ordinary memo lost paid complete original rows"));
        }
        Ok(())
    }
    pub(in crate::margin_row) fn validate_adapter(
        &self,
        adapter: &PreparedRootAdapter,
        requested: Option<Instant>,
    ) -> Result<()> {
        if !self.adapter().same_owner(adapter) {
            return Err(invalid("ordinary memo foreign original projection"));
        }
        self.checkpoint(requested)
    }
    pub(in crate::margin_row) fn box_bounds(
        &self,
        requested: Option<Instant>,
    ) -> Result<(&[f64], &[f64])> {
        self.checkpoint(requested)?;
        Ok((&self.0.lo, &self.0.hi))
    }
    pub(in crate::margin_row) fn lower(
        &self,
        column: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedOrdinaryRowView<'_>> {
        self.view(column, true, requested)
    }
    pub(in crate::margin_row) fn upper(
        &self,
        column: usize,
        requested: Option<Instant>,
    ) -> Result<PreparedOrdinaryRowView<'_>> {
        self.view(column, false, requested)
    }
    fn view(
        &self,
        column: usize,
        lower: bool,
        requested: Option<Instant>,
    ) -> Result<PreparedOrdinaryRowView<'_>> {
        self.checkpoint(requested)?;
        if column >= self.columns() {
            return Err(invalid("ordinary memo column outside original head"));
        }
        Ok(PreparedOrdinaryRowView {
            owner: self,
            planes: if lower { &self.0.lower } else { &self.0.upper },
            column,
        })
    }
    /// Borrowed original-coordinate planes. A root memo/YPack consumer must
    /// keep this owner, and price its own scans, copies and downstream work.
    pub(in crate::margin_row) fn matrices(
        &self,
        lower: bool,
        requested: Option<Instant>,
    ) -> Result<(ArrayView2<'_, f64>, ArrayView2<'_, f64>, &[f64], &[f64])> {
        self.checkpoint(requested)?;
        let p = if lower { &self.0.lower } else { &self.0.upper };
        Ok((p.a.view(), p.e.view(), &p.b, &p.eb))
    }
}

pub(in crate::margin_row) struct PreparedOrdinaryRowView<'a> {
    owner: &'a PreparedOrdinaryRows,
    planes: &'a Planes,
    column: usize,
}
impl PreparedOrdinaryRowView<'_> {
    pub(in crate::margin_row) fn original_output(&self) -> usize {
        self.column
    }
    pub(in crate::margin_row) fn bias(&self) -> f64 {
        self.planes.b[self.column]
    }
    pub(in crate::margin_row) fn bias_error(&self) -> f64 {
        self.planes.eb[self.column]
    }
    /// Read-only exact original input IDs; this iterator allocates nothing.
    /// Arithmetic over it remains the consumer's original-budget obligation.
    pub(in crate::margin_row) fn entries(&self) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        let n = self.owner.n_in();
        (0..n).map(|i| {
            (
                i,
                self.planes.a[[i, self.column]],
                self.planes.e[[i, self.column]],
            )
        })
    }
}

struct Plan {
    persistent: usize,
    output_work: u64,
    tile_fixed: usize,
    tile_per_column: usize,
    work_per_column: u64,
    widest: usize,
}

/// Complete ordinary y rows on the same model/root/query authority. Resource
/// refusal leaves the caller's authentic projection usable for its fallback.
pub(in crate::margin_row) fn build_prepared_ordinary_rows(
    adapter: PreparedRootAdapter,
    requested: Option<Instant>,
) -> Result<PreparedOrdinaryRows> {
    build_inner(adapter, requested, None)
}

// Controls can observe a completed actual tile and cancel its original
// budget. They cannot supply rows, seeds, roots, errors or publication facts.
fn build_inner(
    adapter: PreparedRootAdapter,
    requested: Option<Instant>,
    mut after_tile: Option<&mut dyn FnMut(usize)>,
) -> Result<PreparedOrdinaryRows> {
    let mut at = adapter.deadline(requested)?;
    adapter.checkpoint(Some(at))?;
    let budget = adapter.source().budget();
    qualify(budget, at)?;
    let net = adapter.net();
    // This O(1) bound pays the geometry walk before any operator/vector scan.
    let mut census = admit(
        budget,
        WorkRequest {
            scratch_bytes: META,
            work: work(
                add(add(net.ops.len(), net.tsize.len())?, net.trunk_relus.len())?,
                1024,
            )?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &census);
    let plan = plan(&adapter, at)?;
    census.complete().map_err(refusal)?;
    drop(census);
    let mut owner = admit(
        budget,
        WorkRequest {
            persistent_bytes: plan.persistent,
            work: plan.output_work,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &owner);
    let n = net.n_in;
    let columns = net.n_y;
    let mut lower = zero_planes(n, columns, budget, at)?;
    let mut upper = zero_planes(n, columns, budget, at)?;
    let mut lo = filled(columns, budget, at)?;
    let mut hi = filled(columns, budget, at)?;
    let mut receipt = PreparedOrdinaryPassReceipt::default();
    let mut first = 0;
    while first < columns {
        qualify(budget, at)?;
        let desired = (columns - first).min(MAX_TILE);
        let (mut walk, width) = tile_lease(&plan, budget, desired, at, &mut receipt)?;
        at = min_lease(at, &walk);
        let mut seed = Seed {
            s: Array2::from_shape_vec((columns, width), filled(mul(columns, width)?, budget, at)?)
                .map_err(|_| invalid("ordinary identity tile shape changed"))?,
            e: None,
        };
        for j in 0..width {
            poll(budget, at, j)?;
            seed.s[[first + j, j]] = 1.0;
        }
        // Transfer each direction before producing the other, so the live
        // tile census need not hide two independently retained full passes.
        let out = run_prepared_ordinary_tile(&adapter, &seed, LaneDir::Lower, at)?;
        transfer(out, &mut lower, first, width, budget, at)?;
        receipt.backward_passes = add(receipt.backward_passes, 1)?;
        receipt.lower_columns = add(receipt.lower_columns, width)?;
        let out = run_prepared_ordinary_tile(&adapter, &seed, LaneDir::Upper, at)?;
        transfer(out, &mut upper, first, width, budget, at)?;
        receipt.backward_passes = add(receipt.backward_passes, 1)?;
        receipt.upper_columns = add(receipt.upper_columns, width)?;
        drop(seed);
        walk.complete().map_err(refusal)?;
        drop(walk);
        receipt.tiles = add(receipt.tiles, 1)?;
        if let Some(observe) = after_tile.as_mut() {
            observe(receipt.tiles);
        }
        qualify(budget, at)?;
        first = add(first, width)?;
    }
    concretize(&adapter, &lower, true, &mut lo, at)?;
    concretize(&adapter, &upper, false, &mut hi, at)?;
    for j in 0..columns {
        poll(budget, at, j)?;
        if lo[j] > hi[j] {
            return Err(invalid("ordinary memo inverted concretized output"));
        }
    }
    receipt.stored_coefficient_cells = mul(mul(n, columns)?, 2)?;
    receipt.stored_error_cells = receipt.stored_coefficient_cells;
    receipt.concretized_columns = mul(columns, 2)?;
    // All capacities were reconciled immediately after each allocation and
    // transfer. No numerical owner is published until both passes complete.
    adapter.checkpoint(Some(at))?;
    qualify(budget, at)?;
    owner.mark_materialized().map_err(refusal)?;
    let resident = owner.commit_persistent().map_err(refusal)?;
    let result = PreparedOrdinaryRows(
        Arc::new(OrdinaryInner {
            adapter,
            lower,
            upper,
            lo,
            hi,
            deadline: at,
            receipt,
            resident,
        }),
        at,
    );
    result.checkpoint(Some(at))?;
    ny_core::require_f64_interval_proof_environment()?;
    Ok(result)
}

fn plan(adapter: &PreparedRootAdapter, at: Instant) -> Result<Plan> {
    let net = adapter.net();
    let budget = adapter.source().budget();
    if net.n_in == 0 || net.n_y == 0 || net.ops.len() > MAX_OPS {
        return Err(invalid("ordinary memo unsupported original geometry"));
    }
    let mut rows = 0;
    let mut widest = net.n_y;
    for (i, &n) in net.tsize.iter().enumerate() {
        poll(budget, at, i)?;
        if i <= net.i_gemm1 {
            rows = add(rows, n)?;
            widest = widest.max(n);
        }
    }
    let mut max_ci = 0;
    let mut arithmetic = add(add(mul(net.n_in, 128)?, mul(net.n_y, 64)?)?, 4096)?;
    for (i, op) in net.ops.iter().enumerate() {
        poll(budget, at, i)?;
        arithmetic = add(arithmetic, 512)?;
        match op {
            TwinOp::Conv(c) => {
                arithmetic = add(arithmetic, mul(c.back_taps.len(), 64)?)?;
                if i < net.i_gemm1 {
                    max_ci = max_ci.max(c.kernel.1);
                    arithmetic = add(
                        arithmetic,
                        mul(mul(net.tsize[c.input], add(c.k_bwd, 1)?)?, 64)?,
                    )?;
                }
            }
            TwinOp::Gemm { shape, .. } if i == net.i_gemm1 => {
                arithmetic = add(arithmetic, mul(mul(shape.0, shape.1)?, 256)?)?;
            }
            _ if i < net.i_gemm1 => arithmetic = add(arithmetic, mul(net.tsize[i + 1], 256)?)?,
            _ => {}
        }
    }
    let workers = rayon::current_num_threads();
    // Covers seed, every residual frontier, removed current/destination,
    // coefficient/error temporaries, output raw-vector transfer, and all
    // worker conv accumulators. The full retained memo is priced separately.
    let tile_per_column = add(
        add(mul(rows, 128)?, mul(net.n_y, 16)?)?,
        add(mul(max_ci, mul(workers, 16)?)?, 1024)?,
    )?;
    let tile_fixed = add(
        add(
            mul(
                add(net.ops.len(), 1)?,
                add(
                    mul(size_of::<Option<crate::margin_row::engine::LaneMat>>(), 2)?,
                    32,
                )?,
            )?,
            mul(workers, 8192)?,
        )?,
        32 * 1024,
    )?;
    let cells = mul(net.n_in, net.n_y)?;
    // Attempt bookkeeping is bounded by seven widths per output tile, even
    // when every successful tile contains one column. Paid scans include
    // all four planes, copy/finite checks, concretization and source checks.
    let output_work = work(
        add(
            add(mul(cells, 4)?, mul(net.n_y, 16)?)?,
            mul(
                add(net.ops.len(), net.trunk_relus.len())?,
                add(mul(net.n_y, 2)?, 4)?,
            )?,
        )?,
        256,
    )?;
    Ok(Plan {
        persistent: owner_bytes(net.n_in, net.n_y)?,
        output_work,
        tile_fixed,
        tile_per_column,
        work_per_column: u64::try_from(arithmetic).map_err(|_| overflow())?,
        widest,
    })
}

fn tile_lease(
    plan: &Plan,
    budget: &WorkBudget,
    desired: usize,
    at: Instant,
    receipt: &mut PreparedOrdinaryPassReceipt,
) -> Result<(WorkLease, usize)> {
    let mut width = desired;
    loop {
        qualify(budget, at)?;
        receipt.tile_admission_attempts = add(receipt.tile_admission_attempts, 1)?;
        let cells = mul(plan.widest, width)?;
        if cells > MAX_CELLS {
            if width == 1 {
                return Err(cap_error(cells, MAX_CELLS, "ordinary tile cell policy"));
            }
            width = width.div_ceil(2);
            continue;
        }
        let trial = budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: add(plan.tile_fixed, mul(plan.tile_per_column, width)?)?,
                work: plan
                    .work_per_column
                    .checked_mul(width as u64)
                    .and_then(|w| w.checked_mul(2))
                    .ok_or_else(overflow)?,
                ..WorkRequest::default()
            },
        );
        match trial {
            Ok(lease) => {
                checkpoint(budget, Some(at))?;
                receipt.max_tile_columns = receipt.max_tile_columns.max(width);
                return Ok((lease, width));
            }
            Err(
                WorkRefusal::ScratchLimit | WorkRefusal::PersistentLimit | WorkRefusal::MemoryLimit,
            ) if width > 1 => width = width.div_ceil(2),
            Err(error) => return Err(refusal(error)),
        }
    }
}

fn zero_planes(n: usize, columns: usize, budget: &WorkBudget, at: Instant) -> Result<Planes> {
    let matrix = |v| {
        Array2::from_shape_vec((n, columns), v)
            .map_err(|_| invalid("ordinary memo matrix shape changed"))
    };
    Ok(Planes {
        a: matrix(filled(mul(n, columns)?, budget, at)?)?,
        e: matrix(filled(mul(n, columns)?, budget, at)?)?,
        b: filled(columns, budget, at)?,
        eb: filled(columns, budget, at)?,
    })
}

fn transfer(
    out: PassOut,
    target: &mut Planes,
    first: usize,
    width: usize,
    budget: &WorkBudget,
    at: Instant,
) -> Result<()> {
    let rows = target.a.nrows();
    if out.a.dim() != (rows, width)
        || out.a.as_slice().is_none()
        || out
            .e
            .as_ref()
            .is_none_or(|e| e.dim() != out.a.dim() || e.as_slice().is_none())
        || out.b.len() != width
        || out.eb.len() != width
        || out.coll.is_some()
        || out.coll_rows.is_some()
        || out.unst_rows.is_some()
        || add(first, width)? > target.a.ncols()
    {
        return Err(invalid("ordinary kernel returned incomplete original tile"));
    }
    let PassOut { a, e, b, eb, .. } = out;
    let (a, offset_a) = a.into_raw_vec_and_offset();
    let (e, offset_e) = e.expect("checked").into_raw_vec_and_offset();
    let cells = mul(rows, width)?;
    for (v, offset) in [(&a, offset_a), (&e, offset_e)] {
        if offset.unwrap_or(0) != 0 || v.len() != cells || v.capacity() != cells {
            return Err(cap_error(v.capacity(), cells, "ordinary raw tile capacity"));
        }
    }
    if b.capacity() != width || eb.capacity() != width {
        return Err(cap_error(
            b.capacity().max(eb.capacity()),
            width,
            "ordinary tile scalar capacity",
        ));
    }
    for i in 0..rows {
        for j in 0..width {
            let k = i * width + j;
            poll(budget, at, k)?;
            if !a[k].is_finite() || !e[k].is_finite() || e[k] < 0.0 {
                return Err(invalid("ordinary tile invalid coefficient/error"));
            }
            target.a[[i, first + j]] = a[k];
            target.e[[i, first + j]] = e[k];
        }
    }
    for j in 0..width {
        poll(budget, at, j)?;
        if !b[j].is_finite() || !eb[j].is_finite() || eb[j] < 0.0 {
            return Err(invalid("ordinary tile invalid bias/error"));
        }
        target.b[first + j] = b[j];
        target.eb[first + j] = eb[j];
    }
    checkpoint(budget, Some(at))
}

/// Same operation order and expressions as BackwardEngine::concretize, but
/// writes only into the preadmitted memo and polls the original authority.
fn concretize(
    adapter: &PreparedRootAdapter,
    p: &Planes,
    lower: bool,
    output: &mut [f64],
    at: Instant,
) -> Result<()> {
    concretize_planes(adapter, &p.a, &p.e, &p.b, &p.eb, lower, output, at)
}

/// Arithmetic-only adapter for a genuinely seeded direct tile. The caller
/// must preadmit its output vector and the complete live numerical frontier.
pub(in crate::margin_row) fn concretize_prepared_pass(
    adapter: &PreparedRootAdapter,
    pass: &PassOut,
    lower: bool,
    requested: Instant,
) -> Result<Vec<f64>> {
    let at = adapter.deadline(Some(requested))?;
    adapter.checkpoint(Some(at))?;
    let columns = pass.a.ncols();
    let errors = pass
        .e
        .as_ref()
        .ok_or_else(|| invalid("prepared direct missing errors"))?;
    if pass.a.nrows() != adapter.net().n_in
        || errors.dim() != pass.a.dim()
        || pass.b.len() != columns
        || pass.eb.len() != columns
        || pass.a.as_slice().is_none()
        || errors.as_slice().is_none()
    {
        return Err(invalid("prepared direct incomplete original planes"));
    }
    let mut result = filled(columns, adapter.source().budget(), at)?;
    concretize_planes(
        adapter,
        &pass.a,
        errors,
        &pass.b,
        &pass.eb,
        lower,
        &mut result,
        at,
    )?;
    adapter.checkpoint(Some(at))?;
    Ok(result)
}

/// Geometry-only cached plan. Its caller must admit the bounded original
/// metadata walk before construction. Requests afterward are O(1); no source
/// or numerical authority is minted by this arithmetic resource estimate.
pub(in crate::margin_row) struct PreparedOrdinaryWalkPlan {
    tile_fixed: usize,
    tile_per_column: usize,
    work_per_column: u64,
    widest: usize,
}
pub(in crate::margin_row) fn prepared_ordinary_walk_plan(
    adapter: &PreparedRootAdapter,
    requested: Instant,
) -> Result<PreparedOrdinaryWalkPlan> {
    let at = adapter.deadline(Some(requested))?;
    adapter.checkpoint(Some(at))?;
    let p = plan(adapter, at)?;
    Ok(PreparedOrdinaryWalkPlan {
        tile_fixed: p.tile_fixed,
        tile_per_column: p.tile_per_column,
        work_per_column: p.work_per_column,
        widest: p.widest,
    })
}
impl PreparedOrdinaryWalkPlan {
    pub(in crate::margin_row) fn request(&self, columns: usize) -> Result<WorkRequest> {
        if columns == 0 || columns > MAX_TILE {
            return Err(invalid("prepared direct unsupported tile width"));
        }
        let cells = mul(self.widest, columns)?;
        if cells > MAX_CELLS {
            return Err(cap_error(cells, MAX_CELLS, "prepared direct tile cells"));
        }
        Ok(WorkRequest {
            scratch_bytes: add(self.tile_fixed, mul(self.tile_per_column, columns)?)?,
            work: self
                .work_per_column
                .checked_mul(columns as u64)
                .and_then(|w| w.checked_mul(2))
                .ok_or_else(overflow)?,
            ..WorkRequest::default()
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn concretize_planes(
    adapter: &PreparedRootAdapter,
    coefficients: &Array2<f64>,
    errors: &Array2<f64>,
    bias: &[f64],
    bias_errors: &[f64],
    lower: bool,
    output: &mut [f64],
    at: Instant,
) -> Result<()> {
    let root = adapter.gates();
    let budget = adapter.source().budget();
    let n = root.mid.len();
    let gam = gamma_n(add(n, 16)?);
    if gam > 0.25 {
        return Err(invalid(
            "ordinary concretization proof length exceeds envelope",
        ));
    }
    for j in 0..output.len() {
        let (mut v, mut rr, mut tabs, mut pen, mut dust) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            poll(budget, at, i)?;
            let a = coefficients[[i, j]];
            v += a * root.mid[i];
            let aa = a.abs();
            rr += aa * root.rad[i];
            tabs += error_mul(aa, root.xabs[i], true);
            let dm = nominal_product_dust(a, root.mid[i]);
            let dr = nominal_product_dust(aa, root.rad[i]);
            if dm != 0.0 {
                dust = next_up(dust + dm);
            }
            if dr != 0.0 {
                dust = next_up(dust + dr);
            }
            pen += error_mul(errors[[i, j]], root.xabs[i], true);
        }
        let base = v + bias[j];
        let raw = if lower { base - rr } else { base + rr };
        let mut slack = next_up(
            certify_up(pen + bias_errors[j], 1e-14)
                + next_up(error_mul(
                    gam,
                    tabs + bias[j].abs() + pen + bias_errors[j],
                    true,
                )),
        );
        if dust != 0.0 {
            slack = next_up(slack + certify_up(dust, gam));
        }
        output[j] = if lower {
            next_down(next_down(raw - slack))
        } else {
            next_up(next_up(raw + slack))
        };
        if !output[j].is_finite() {
            return Err(invalid("ordinary concretization nonfinite"));
        }
    }
    qualify(budget, at)
}

fn owner_bytes(n: usize, columns: usize) -> Result<usize> {
    add(
        mul(add(mul(mul(n, columns)?, 4)?, mul(columns, 6)?)?, 8)?,
        add(
            add(size_of::<OrdinaryInner>(), mul(size_of::<usize>(), 2)?)?,
            META,
        )?,
    )
}
fn filled(n: usize, budget: &WorkBudget, at: Instant) -> Result<Vec<f64>> {
    checkpoint(budget, Some(at))?;
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| cap_error(n, n, "ordinary allocation failed"))?;
    checkpoint(budget, Some(at))?;
    if v.capacity() != n {
        return Err(cap_error(v.capacity(), n, "ordinary actual capacity"));
    }
    for i in 0..n {
        poll(budget, at, i)?;
        v.push(0.0);
    }
    checkpoint(budget, Some(at))?;
    Ok(v)
}
fn admit(budget: &WorkBudget, request: WorkRequest, at: Instant) -> Result<WorkLease> {
    qualify(budget, at)?;
    let lease = budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(refusal)?;
    checkpoint(budget, Some(at))?;
    Ok(lease)
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| at.min(d))
}
fn qualify(budget: &WorkBudget, at: Instant) -> Result<()> {
    checkpoint(budget, Some(at))?;
    ny_core::require_f64_interval_proof_environment()
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, Some(at))?;
    }
    Ok(())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn work(a: usize, b: usize) -> Result<u64> {
    u64::try_from(mul(a, b)?).map_err(|_| overflow())
}
fn overflow() -> NyError {
    invalid("ordinary row resource geometry overflow")
}
fn cap_error(requested: usize, limit: usize, label: &'static str) -> NyError {
    NyError::CpuMemoryExceeded {
        required_bytes: requested,
        budget_bytes: limit,
        site: label,
    }
}

#[cfg(test)]
#[path = "ordinary_pass_tests.rs"]
mod tests;
