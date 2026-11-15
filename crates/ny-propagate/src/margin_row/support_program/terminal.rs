// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete original-head boxes and margins from a completed prepared trunk.
//!
//! Identity head tiles and original margin tiles use the existing certified
//! CPU backward kernel. No output-by-input identity pack is retained. Tile
//! width is admitted before seed allocation and may shrink to one column.
//! The final owner binds the immutable model/root/prefix, the actual original
//! Gemm1 input and ordered class mapping. It is numerical evidence only; CLI
//! property agreement, containment and the common cover authorizer remain
//! separate mandatory publication work.
//!
//! Final-Gemm differences are not assumed exact: TwoSum residuals are charged
//! over the complete head box. ReLU upper lines enclose their actual endpoints;
//! signed seed products, intercepts, concretization and every error product are
//! rounded outwards, including gradual-underflow losses. Head/trunk fold and
//! coefficient errors remain the original backward kernel's authority.

use super::construction::PreparedRootConstruction;
use super::ir::{checkpoint, invalid, refusal};
use super::prefix::PreparedPrefix;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest};
use crate::margin_row::engine::{run_prepared_terminal_tile, LaneDir, PassOut, Seed};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::net::TwinOp;
use crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
use ndarray::Array2;
use ny_core::dd::{next_down_f64, next_up_f64, two_sum};
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

const MAX_TILE: usize = 64;
const MAX_CELLS: usize = 16 * 1024 * 1024;
const POLL: usize = 256;
const OWNER: usize = 1024;

/// Completed numerical observations, not a proof token or RSS measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::margin_row) struct PreparedTerminalReceipt {
    pub head_lower_columns: usize,
    pub head_upper_columns: usize,
    pub margin_columns: usize,
    pub head_tiles: usize,
    pub margin_tiles: usize,
    pub backward_passes: usize,
    pub max_tile_columns: usize,
    pub tile_admission_attempts: usize,
    pub parameter_difference_residual_terms: usize,
    pub seed_rounding_terms: usize,
}

struct TerminalInner {
    prefix: PreparedPrefix,
    head_input: usize,
    target: usize,
    adversaries: Vec<usize>,
    head_lo: Vec<f64>,
    head_hi: Vec<f64>,
    lower: Vec<f64>,
    deadline: Instant,
    receipt: PreparedTerminalReceipt,
    resident: PersistentReservation,
}

/// Cloning shares the exact immutable evidence and one last-owner byte claim.
/// There is no supplied-numeric, deserializing or mutable constructor.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedTerminal(Arc<TerminalInner>);

impl PreparedTerminal {
    pub(in crate::margin_row) fn prefix(&self) -> &PreparedPrefix {
        &self.0.prefix
    }
    pub(in crate::margin_row) fn head_input(&self) -> usize {
        self.0.head_input
    }
    pub(in crate::margin_row) fn target(&self) -> usize {
        self.0.target
    }
    pub(in crate::margin_row) fn adversaries(&self) -> &[usize] {
        &self.0.adversaries
    }
    pub(in crate::margin_row) fn head_lo(&self) -> &[f64] {
        &self.0.head_lo
    }
    pub(in crate::margin_row) fn head_hi(&self) -> &[f64] {
        &self.0.head_hi
    }
    pub(in crate::margin_row) fn lower(&self) -> &[f64] {
        &self.0.lower
    }
    pub(in crate::margin_row) fn receipt(&self) -> PreparedTerminalReceipt {
        self.0.receipt
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        let at = requested.map_or(self.0.deadline, |d| d.min(self.0.deadline));
        self.prefix()
            .deadline(Some(at))?
            .ok_or_else(|| invalid("missing terminal deadline"))
    }

    /// The publisher must keep this actual owner and recheck after all guards.
    /// Equal hashes/limits or another completed frame do not confer authority.
    pub(in crate::margin_row) fn validate_source(
        &self,
        source: &PreparedRootConstruction,
        target: usize,
        adversaries: &[usize],
        requested: Option<Instant>,
    ) -> Result<()> {
        let mut at = self.deadline(requested)?.min(source.deadline(requested)?);
        qualify(self.prefix().root().budget(), at)?;
        let mut validation = admit(
            self.prefix().root().budget(),
            WorkRequest {
                work: charged(self.adversaries().len(), 16)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        at = min_lease(at, &validation);
        if !self.prefix().same_prefix(source.prefix())
            || target != self.target()
            || adversaries.len() != self.adversaries().len()
            || !self.0.resident.shares_ledger(self.prefix().root().budget())
            || source.trunk_output().tensor() != self.head_input()
            || !source.trunk_output().root().same_root(self.prefix().root())
            || self.head_lo().len() != self.prefix().root().model().net().n_y
            || self.head_hi().len() != self.head_lo().len()
            || self.lower().len() != self.adversaries().len()
        {
            return Err(invalid(
                "foreign/stale prepared terminal source or objective",
            ));
        }
        for (i, (&requested, &sealed)) in adversaries.iter().zip(self.adversaries()).enumerate() {
            poll(self.prefix().root().budget(), at, i)?;
            if requested != sealed {
                return Err(invalid("prepared terminal original class order changed"));
            }
        }
        qualify(self.prefix().root().budget(), at)?;
        validation.complete().map_err(refusal)
    }
}

#[derive(Clone, Copy)]
struct HeadLine {
    alpha: f64,
    slope: f64,
    intercept: f64,
}

struct Plan {
    persistent: usize,
    scratch: usize,
    work: u64,
    walk_bytes_per_column: usize,
    walk_work_per_column: u64,
    walk_fixed_bytes: usize,
    frame_bytes: usize,
    widest: usize,
    head_input: usize,
}

pub(in crate::margin_row) fn build_prepared_terminal(
    source: &PreparedRootConstruction,
    target: usize,
    adversaries: &[usize],
    requested: Option<Instant>,
) -> Result<PreparedTerminal> {
    build_inner(source, target, adversaries, requested, None)
}

// An observer is only used by controls to interrupt actual completed tiles.
fn build_inner(
    source: &PreparedRootConstruction,
    target: usize,
    adversaries: &[usize],
    requested: Option<Instant>,
    mut observer: Option<&mut dyn FnMut(&PreparedTerminalReceipt)>,
) -> Result<PreparedTerminal> {
    let prefix = source.prefix();
    let root = prefix.root();
    let budget = root.budget();
    let net = root.model().net();
    let mut at = source.deadline(requested)?;
    at = root
        .deadline(source.trunk_output().deadline()?.map(|d| d.min(at)))?
        .ok_or_else(|| invalid("missing terminal source clock"))?
        .min(at);
    qualify(budget, at)?;
    if root.policy().err_lane != ErrLane::PerCol
        || prefix.layer_count() != net.trunk_relus.len()
        || net.n_y == 0
        || target >= net.n_out
        || adversaries.is_empty()
        || adversaries.len() >= net.n_out
    {
        return Err(invalid(
            "unsupported terminal source or empty/out-of-range class mapping",
        ));
    }
    // Price original geometry inspection before traversing any DAG/parameters.
    let mut header = admit(
        budget,
        WorkRequest {
            scratch_bytes: add(OWNER, mul(adversaries.len(), size_of::<usize>())?)?,
            work: charged(
                add(add(mul(net.ops.len(), 512)?, adversaries.len())?, net.n_out)?,
                128,
            )?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &header);
    let plan = plan(source, adversaries.len(), at)?;
    header.complete().map_err(refusal)?;
    drop(header);
    let mut whole = admit(
        budget,
        WorkRequest {
            scratch_bytes: plan.scratch,
            persistent_bytes: plan.persistent,
            work: plan.work,
            proposals: 0,
        },
        at,
    )?;
    at = min_lease(at, &whole);
    let words = add(net.n_out, 63)? / 64;
    let mut seen = filled(words, 0u64, budget, at)?;
    let mut classes = exact_vec(adversaries.len(), budget, at)?;
    for (i, &class) in adversaries.iter().enumerate() {
        poll(budget, at, i)?;
        if class >= net.n_out || class == target || seen[class / 64] & (1u64 << (class % 64)) != 0 {
            return Err(invalid("invalid/duplicate terminal original adversary"));
        }
        seen[class / 64] |= 1u64 << (class % 64);
        classes.push(class);
    }
    let mut head_lo = filled(net.n_y, 0.0, budget, at)?;
    let mut head_hi = filled(net.n_y, 0.0, budget, at)?;
    let mut lower = filled(classes.len(), 0.0, budget, at)?;
    let mut receipt = PreparedTerminalReceipt::default();
    whole.mark_materialized().map_err(refusal)?;
    let mut start = 0usize;
    while start < net.n_y {
        qualify(budget, at)?;
        let (mut tile, columns) = tile_lease(
            &plan,
            source,
            (net.n_y - start).min(MAX_TILE),
            2,
            at,
            &mut receipt,
        )?;
        at = min_lease(at, &tile);
        let mut seed = zero_seed(net.n_y, columns, false, budget, at)?;
        for column in 0..columns {
            seed.s[[start + column, column]] = 1.0;
        }
        tile.mark_materialized().map_err(refusal)?;
        for upper in [false, true] {
            let pass = run_prepared_terminal_tile(
                source,
                &seed,
                if upper {
                    LaneDir::Upper
                } else {
                    LaneDir::Lower
                },
                at,
            )?;
            concretize(
                &pass,
                root.lo(),
                root.hi(),
                upper,
                if upper {
                    &mut head_hi[start..start + columns]
                } else {
                    &mut head_lo[start..start + columns]
                },
                budget,
                at,
            )?;
            drop(pass);
            receipt.backward_passes = add(receipt.backward_passes, 1)?;
            if upper {
                receipt.head_upper_columns = add(receipt.head_upper_columns, columns)?;
            } else {
                receipt.head_lower_columns = add(receipt.head_lower_columns, columns)?;
            }
            notify(&mut observer, &receipt);
            qualify(budget, at)?;
        }
        drop(seed);
        tile.complete().map_err(refusal)?;
        drop(tile);
        receipt.head_tiles = add(receipt.head_tiles, 1)?;
        start += columns;
    }
    let mut lines = exact_vec(net.n_y, budget, at)?;
    for j in 0..net.n_y {
        poll(budget, at, j)?;
        lines.push(head_line(head_lo[j], head_hi[j])?);
    }
    let (weights, bias, _) = net.gemm2();
    start = 0;
    while start < classes.len() {
        qualify(budget, at)?;
        let (mut tile, columns) = tile_lease(
            &plan,
            source,
            (classes.len() - start).min(MAX_TILE),
            1,
            at,
            &mut receipt,
        )?;
        at = min_lease(at, &tile);
        let mut seed = zero_seed(net.n_y, columns, true, budget, at)?;
        let mut offsets = exact_vec(columns, budget, at)?;
        let mut interval = exact_vec(columns, budget, at)?;
        for (column, &class) in classes[start..start + columns].iter().enumerate() {
            let (constant, constant_error) = difference(bias[target], bias[class])?;
            if constant_error != 0.0 {
                receipt.parameter_difference_residual_terms =
                    add(receipt.parameter_difference_residual_terms, 1)?;
            }
            let mut offset = sub_down(constant, constant_error)?;
            let mut ibp = offset;
            for j in 0..net.n_y {
                poll(budget, at, j)?;
                let (q, error) =
                    difference(weights[target * net.n_y + j], weights[class * net.n_y + j])?;
                if error != 0.0 {
                    receipt.parameter_difference_residual_terms =
                        add(receipt.parameter_difference_residual_terms, 1)?;
                }
                let line = lines[j];
                let (coefficient, coefficient_error) =
                    product_center_error(q, if q >= 0.0 { line.alpha } else { line.slope })?;
                seed.s[[j, column]] = coefficient;
                seed.e.as_mut().expect("margin error plane")[[j, column]] = coefficient_error;
                if coefficient_error != 0.0 {
                    receipt.seed_rounding_terms = add(receipt.seed_rounding_terms, 1)?;
                }
                if q < 0.0 {
                    offset = add_down(offset, mul_down(q, line.intercept)?)?;
                }
                let zlo = head_lo[j].max(0.0);
                let zhi = head_hi[j].max(0.0);
                let residual = mul_up(error, zhi)?;
                offset = sub_down(offset, residual)?;
                ibp = sub_down(
                    add_down(ibp, mul_down(q, if q >= 0.0 { zlo } else { zhi })?)?,
                    residual,
                )?;
            }
            offsets.push(offset);
            interval.push(ibp);
        }
        tile.mark_materialized().map_err(refusal)?;
        let pass = run_prepared_terminal_tile(source, &seed, LaneDir::Lower, at)?;
        concretize(
            &pass,
            root.lo(),
            root.hi(),
            false,
            &mut lower[start..start + columns],
            budget,
            at,
        )?;
        for column in 0..columns {
            poll(budget, at, column)?;
            lower[start + column] =
                add_down(lower[start + column], offsets[column])?.max(interval[column]);
        }
        drop(pass);
        drop(seed);
        drop(offsets);
        drop(interval);
        tile.complete().map_err(refusal)?;
        drop(tile);
        receipt.margin_columns = add(receipt.margin_columns, columns)?;
        receipt.margin_tiles = add(receipt.margin_tiles, 1)?;
        receipt.backward_passes = add(receipt.backward_passes, 1)?;
        notify(&mut observer, &receipt);
        qualify(budget, at)?;
        start += columns;
    }
    if receipt.head_lower_columns != net.n_y
        || receipt.head_upper_columns != net.n_y
        || receipt.margin_columns != classes.len()
    {
        return Err(invalid("incomplete prepared terminal delivery"));
    }
    let actual = add(
        add(
            mul(add(head_lo.capacity(), head_hi.capacity())?, 8)?,
            mul(lower.capacity(), 8)?,
        )?,
        add(mul(classes.capacity(), size_of::<usize>())?, owner_bytes())?,
    )?;
    if actual > plan.persistent {
        return Err(cap_error(
            actual,
            plan.persistent,
            "terminal allocator capacity",
        ));
    }
    drop(lines);
    drop(seen);
    qualify(budget, at)?;
    let resident = whole.commit_persistent().map_err(refusal)?;
    let result = PreparedTerminal(Arc::new(TerminalInner {
        prefix: prefix.clone(),
        head_input: plan.head_input,
        target,
        adversaries: classes,
        head_lo,
        head_hi,
        lower,
        deadline: at,
        receipt,
        resident,
    }));
    result.validate_source(source, target, adversaries, Some(at))?;
    Ok(result)
}

fn notify(
    observer: &mut Option<&mut dyn FnMut(&PreparedTerminalReceipt)>,
    receipt: &PreparedTerminalReceipt,
) {
    if let Some(f) = observer.as_mut() {
        f(receipt);
    }
}

fn plan(source: &PreparedRootConstruction, classes: usize, at: Instant) -> Result<Plan> {
    let root = source.prefix().root();
    let budget = root.budget();
    let net = root.model().net();
    if net.ops.len() > 4096 {
        return Err(invalid(
            "terminal original operator count exceeds bounded kernel",
        ));
    }
    let (head_input, head_width) = match net.ops.get(net.i_gemm1) {
        Some(TwinOp::Gemm { input, shape, .. }) if shape.0 == net.n_y => (*input, shape.1),
        _ => return Err(invalid("terminal missing actual original Gemm1")),
    };
    if source.trunk_output().tensor() != head_input
        || !source.trunk_output().root().same_root(root)
        || source.trunk_output().lo().len() != head_width
        || source.trunk_output().hi().len() != head_width
    {
        return Err(invalid("terminal original head-input authority mismatch"));
    }
    if net.ops.len() != add(net.i_gemm1, 3)?
        || !matches!(net.ops.get(net.i_gemm1+1), Some(TwinOp::Relu { input, layer }) if *input == net.i_gemm1+1 && *layer == usize::MAX)
        || !matches!(net.ops.last(), Some(TwinOp::Gemm { input, weight_rel_err, shape, .. })
            if *input == net.i_gemm1+2 && *weight_rel_err == 0.0 && *shape == (net.n_out, net.n_y))
    {
        return Err(invalid(
            "terminal original head tail/final exactness mismatch",
        ));
    }
    if let Some(TwinOp::Gemm { bias_err, .. }) = net.ops.last() {
        for (i, &error) in bias_err.iter().enumerate() {
            poll(budget, at, i)?;
            if error != 0.0 {
                return Err(invalid("terminal original final Gemm has bias uncertainty"));
            }
        }
    }
    let mut rows = 0usize;
    let mut widest = 0usize;
    for (i, &n) in net.tsize.iter().enumerate() {
        poll(budget, at, i)?;
        if i <= net.i_gemm1 {
            rows = add(rows, n)?;
            widest = widest.max(n);
        }
    }
    let mut operations = add(mul(mul(net.n_y, head_width)?, 512)?, mul(net.n_in, 128)?)?;
    let mut fixed_work = 4096usize;
    let mut max_ci = 0usize;
    for (k, op) in net.ops.iter().enumerate() {
        poll(budget, at, k)?;
        fixed_work = add(fixed_work, 1024)?;
        match op {
            TwinOp::Conv(c) => {
                fixed_work = add(fixed_work, mul(c.back_taps.len(), 128)?)?;
                if k < net.i_gemm1 {
                    max_ci = max_ci.max(c.kernel.1);
                    operations = add(
                        operations,
                        mul(mul(net.tsize[c.input], add(c.k_bwd, 1)?)?, 256)?,
                    )?;
                }
            }
            _ if k < net.i_gemm1 => operations = add(operations, mul(net.tsize[k + 1], 512)?)?,
            _ => {}
        }
    }
    operations = add(operations, fixed_work)?;
    // Includes all residual states, removed current/destination/error scratch,
    // actual output A/E planes, full seed A/E and scalar tile constants.
    let walk_bytes_per_column = add(
        add(mul(rows, 64)?, mul(net.n_y, 24)?)?,
        add(mul(max_ci, mul(rayon::current_num_threads(), 8)?)?, 256)?,
    )?;
    let frame_bytes = add(
        mul(
            source.prefix().layer_count(),
            size_of::<&crate::margin_row::root::LayerGates>(),
        )?,
        size_of::<super::prefix::PreparedPrefixFrame<'_>>(),
    )?;
    let walk_fixed_bytes = add(
        add(
            mul(
                add(net.ops.len(), 1)?,
                size_of::<Option<crate::margin_row::engine::LaneMat>>(),
            )?,
            mul(rayon::current_num_threads(), 4096)?,
        )?,
        add(mul(frame_bytes, 2)?, 16 * 1024)?,
    )?;
    let persistent = add(
        add(
            mul(add(mul(net.n_y, 2)?, classes)?, 8)?,
            mul(classes, size_of::<usize>())?,
        )?,
        owner_bytes(),
    )?;
    widest = widest.max(net.n_y);
    let scratch = add(
        add(
            add(
                mul(net.n_y, size_of::<HeadLine>())?,
                mul(add(net.n_out, 63)? / 64, 8)?,
            )?,
            mul(classes, size_of::<usize>())?,
        )?,
        OWNER,
    )?;
    let work = charged(
        add(
            add(mul(net.n_y, classes)?, net.n_y)?,
            add(classes, add(net.n_out, 63)? / 64)?,
        )?,
        1024,
    )?;
    Ok(Plan {
        persistent,
        scratch,
        work,
        walk_bytes_per_column,
        walk_work_per_column: u64::try_from(operations).map_err(|_| overflow())?,
        walk_fixed_bytes,
        frame_bytes,
        widest,
        head_input,
    })
}

fn tile_lease(
    plan: &Plan,
    source: &PreparedRootConstruction,
    desired: usize,
    passes: u64,
    at: Instant,
    receipt: &mut PreparedTerminalReceipt,
) -> Result<(WorkLease, usize)> {
    let budget = source.prefix().root().budget();
    let mut columns = desired;
    loop {
        qualify(budget, at)?;
        receipt.tile_admission_attempts = add(receipt.tile_admission_attempts, 1)?;
        let scratch = add(
            plan.walk_fixed_bytes,
            mul(plan.walk_bytes_per_column, columns)?,
        )?;
        let request = WorkRequest {
            scratch_bytes: scratch,
            work: plan
                .walk_work_per_column
                .checked_mul(columns as u64)
                .and_then(|n| n.checked_mul(passes))
                .ok_or_else(overflow)?,
            ..WorkRequest::default()
        };
        let local = if mul(plan.widest, columns)? > MAX_CELLS {
            Err(cap_error(
                mul(plan.widest, columns)?,
                MAX_CELLS,
                "terminal tile cell policy",
            ))
        } else {
            preflight(
                budget,
                add(scratch, plan.frame_bytes)?,
                "terminal complete tile overlap",
            )
        };
        let trial = local.and_then(|()| {
            budget
                .lease_for(LeaseClass::Fallback, request)
                .map_err(refusal)
        });
        match trial {
            Ok(lease) => {
                receipt.max_tile_columns = receipt.max_tile_columns.max(columns);
                return Ok((lease, columns));
            }
            Err(error) if columns > 1 => {
                // No tile buffers have been allocated. Every admitted header
                // and refused attempt remains on the original ledger.
                qualify(budget, at)?;
                columns = columns.div_ceil(2);
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    }
}

fn zero_seed(
    heads: usize,
    columns: usize,
    errors: bool,
    budget: &WorkBudget,
    at: Instant,
) -> Result<Seed> {
    let len = mul(heads, columns)?;
    let nominal = Array2::from_shape_vec((heads, columns), filled(len, 0.0, budget, at)?)
        .map_err(|_| invalid("terminal seed shape changed"))?;
    let error = if errors {
        Some(
            Array2::from_shape_vec((heads, columns), filled(len, 0.0, budget, at)?)
                .map_err(|_| invalid("terminal seed error shape changed"))?,
        )
    } else {
        None
    };
    Ok(Seed {
        s: nominal,
        e: error,
    })
}

/// Direct outward support of actual returned coefficients over the DECLARED
/// box. Every nominal product/add and E*xabs is enclosed individually; no
/// uncharged midpoint/radius reconstruction or calibrated scalar slack.
fn concretize(
    pass: &PassOut,
    lo: &[f64],
    hi: &[f64],
    upper: bool,
    out: &mut [f64],
    budget: &WorkBudget,
    at: Instant,
) -> Result<()> {
    let columns = out.len();
    let error = pass
        .e
        .as_ref()
        .ok_or_else(|| invalid("terminal missing certified coefficient error"))?;
    if pass.a.dim() != (lo.len(), columns)
        || error.dim() != pass.a.dim()
        || hi.len() != lo.len()
        || pass.b.len() != columns
        || pass.eb.len() != columns
        || pass.a.as_slice().is_none()
        || error.as_slice().is_none()
    {
        return Err(invalid("terminal kernel output shape/layout mismatch"));
    }
    for column in 0..columns {
        qualify(budget, at)?;
        let mut nominal = finite(pass.b[column])?;
        let mut penalty = nonnegative(pass.eb[column])?;
        for i in 0..lo.len() {
            poll(budget, at, i)?;
            let a = finite(pass.a[[i, column]])?;
            let e = nonnegative(error[[i, column]])?;
            let endpoint = if (a >= 0.0) == upper { hi[i] } else { lo[i] };
            nominal = if upper {
                add_up(nominal, mul_up(a, endpoint)?)?
            } else {
                add_down(nominal, mul_down(a, endpoint)?)?
            };
            penalty = add_up(penalty, mul_up(e, lo[i].abs().max(hi[i].abs()))?)?;
        }
        out[column] = if upper {
            add_up(nominal, penalty)?
        } else {
            sub_down(nominal, penalty)?
        };
    }
    qualify(budget, at)
}

fn head_line(lo: f64, hi: f64) -> Result<HeadLine> {
    finite(lo)?;
    finite(hi)?;
    if lo > hi {
        return Err(invalid("crossed terminal head enclosure"));
    }
    if lo >= 0.0 {
        return Ok(HeadLine {
            alpha: 1.0,
            slope: 1.0,
            intercept: 0.0,
        });
    }
    if hi <= 0.0 {
        return Ok(HeadLine {
            alpha: 0.0,
            slope: 0.0,
            intercept: 0.0,
        });
    }
    let span = hi - lo;
    let mut slope = hi / span;
    if !slope.is_finite() || !(0.0..=1.0).contains(&slope) || !span.is_finite() {
        slope = if -lo >= hi {
            let r = hi / (-lo);
            r / (1.0 + r)
        } else {
            let r = (-lo) / hi;
            1.0 / (1.0 + r)
        };
    }
    // The true endpoint terms are bounded by -lo / hi. Cap an outward
    // overflow at those independently exact bounds before the finite check.
    let left = if slope == 0.0 {
        0.0
    } else {
        next_up_f64(slope * (-lo)).min(-lo)
    };
    let product_lower = if slope == 0.0 {
        0.0
    } else {
        next_down_f64(slope * hi).max(0.0)
    };
    let right = if slope == 1.0 {
        0.0
    } else {
        next_up_f64(hi - product_lower).min(hi)
    };
    let intercept = left.max(right).max(0.0);
    finite(slope)?;
    finite(intercept)?;
    Ok(HeadLine {
        alpha: if hi >= -lo { 1.0 } else { 0.0 },
        slope,
        intercept,
    })
}

fn difference(a: f64, b: f64) -> Result<(f64, f64)> {
    let (nominal, residual) = two_sum(a, -b);
    Ok((finite(nominal)?, nonnegative(residual.abs())?))
}
fn product_center_error(a: f64, b: f64) -> Result<(f64, f64)> {
    let nominal = finite(a * b)?;
    if a == 0.0 || b == 0.0 || b == 1.0 {
        return Ok((nominal, 0.0));
    }
    // Adjacent representable values enclose the true product even if it
    // underflowed. TwoSum pays complete differences from those endpoints.
    let low = finite(next_down_f64(nominal))?;
    let high = finite(next_up_f64(nominal))?;
    let gap = |x: f64| -> Result<f64> {
        let (value, residual) = two_sum(nominal, -x);
        add_up(value.abs(), residual.abs())
    };
    Ok((nominal, gap(low)?.max(gap(high)?)))
}
fn finite(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("nonfinite terminal numerical value"))
    }
}
fn nonnegative(x: f64) -> Result<f64> {
    let x = finite(x)?;
    if x >= 0.0 {
        Ok(x)
    } else {
        Err(invalid("negative terminal error"))
    }
}
fn add_up(a: f64, b: f64) -> Result<f64> {
    if b == 0.0 {
        return finite(a);
    }
    if a == 0.0 {
        return finite(b);
    }
    finite(next_up_f64(a + b))
}
fn add_down(a: f64, b: f64) -> Result<f64> {
    if b == 0.0 {
        return finite(a);
    }
    if a == 0.0 {
        return finite(b);
    }
    finite(next_down_f64(a + b))
}
fn sub_down(a: f64, b: f64) -> Result<f64> {
    add_down(a, -b)
}
fn mul_up(a: f64, b: f64) -> Result<f64> {
    if a == 0.0 || b == 0.0 {
        return Ok(a * b);
    }
    finite(next_up_f64(a * b))
}
fn mul_down(a: f64, b: f64) -> Result<f64> {
    if a == 0.0 || b == 0.0 {
        return Ok(a * b);
    }
    finite(next_down_f64(a * b))
}
fn qualify(budget: &WorkBudget, at: Instant) -> Result<()> {
    checkpoint(budget, Some(at))?;
    ny_core::require_f64_interval_proof_environment()?;
    if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
        return Err(invalid("terminal TwoSum environment refused"));
    }
    Ok(())
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, Some(at))?;
    }
    Ok(())
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| d.min(at))
}
fn owner_bytes() -> usize {
    size_of::<TerminalInner>() + 2 * size_of::<usize>() + OWNER
}
fn overflow() -> NyError {
    invalid("terminal capacity/work overflow")
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn charged(n: usize, factor: usize) -> Result<u64> {
    u64::try_from(add(mul(n, factor)?, OWNER)?).map_err(|_| overflow())
}
fn cap_error(required_bytes: usize, budget_bytes: usize, site: &'static str) -> NyError {
    NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site,
    }
}
fn preflight(budget: &WorkBudget, new: usize, site: &'static str) -> Result<()> {
    let r = budget.receipt().map_err(refusal)?;
    let required = add(add(r.live_scratch_bytes, r.live_persistent_bytes)?, new)?;
    if required > ROOT_SOURCE_BYTES_CAP {
        Err(cap_error(required, ROOT_SOURCE_BYTES_CAP, site))
    } else {
        Ok(())
    }
}
fn admit(budget: &WorkBudget, request: WorkRequest, at: Instant) -> Result<WorkLease> {
    qualify(budget, at)?;
    preflight(
        budget,
        add(request.scratch_bytes, request.persistent_bytes)?,
        "terminal source total live",
    )?;
    budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(refusal)
}
fn exact_vec<T>(n: usize, budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    checkpoint(budget, Some(at))?;
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| invalid("terminal allocation refused"))?;
    checkpoint(budget, Some(at))?;
    if v.capacity() != n {
        return Err(cap_error(
            mul(v.capacity(), size_of::<T>())?,
            mul(n, size_of::<T>())?,
            "terminal exact allocation capacity",
        ));
    }
    Ok(v)
}
fn filled<T: Copy>(n: usize, value: T, budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    let mut v = exact_vec(n, budget, at)?;
    for i in 0..n {
        poll(budget, at, i)?;
        v.push(value);
    }
    Ok(v)
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
