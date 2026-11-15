// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complete paid conditional CPU rows and original-class evaluation. These
//! owners bind actual phase regions; they cannot impersonate unsplit root rows.

use super::*;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use crate::margin_row::bounds::{
    compose_viay_request, compose_viay_with_checkpoint, direct_from_lower_request,
    head_gates_request, head_gates_with_checkpoint, margin_batch_request,
    margin_batch_with_checkpoint, margin_seed_request, margin_seed_with_checkpoint,
    per_class_direct_from_lower_with_checkpoint, row_dots_request, row_dots_with_checkpoint,
};
use crate::margin_row::clip::{
    dual_lower_outward_request, dual_lower_outward_with_checkpoint, Halfspace,
};
use crate::margin_row::engine::{run_prepared_domain_tile, LaneDir};
use crate::margin_row::support_program::{
    concretize_prepared_pass, prepared_ordinary_walk_plan, PreparedDomain, PreparedDomainFrame,
    PreparedOrdinaryWalkPlan, PreparedPhase, PreparedPhaseSide,
};
use ny_core::dd::two_sum;
use std::mem::size_of;

const META: usize = 4096;
const TILE: usize = 64;
const POLL: usize = 256;

struct RowsInner {
    lower: PassOut,
    upper: PassOut,
    lo: Vec<f64>,
    hi: Vec<f64>,
    frame: PreparedDomainFrame,
    plan: PreparedOrdinaryWalkPlan,
    deadline: Instant,
    tiles: usize,
    resident: PersistentReservation,
}

#[derive(Clone)]
pub(in crate::margin_row) struct PreparedDomainRows(Arc<RowsInner>, Instant);
impl PreparedDomainRows {
    pub(in crate::margin_row) fn domain(&self) -> &PreparedDomain {
        self.0.frame.domain()
    }
    pub(in crate::margin_row) fn frame(&self) -> &PreparedDomainFrame {
        &self.0.frame
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0
            .frame
            .deadline(Some(requested.map_or(self.1.min(self.0.deadline), |d| {
                d.min(self.1).min(self.0.deadline)
            })))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.frame.checkpoint(Some(at))?;
        if !self.0.resident.shares_ledger(self.domain().budget())
            || self.0.resident.bytes()
                < rows_bytes(
                    self.domain().adapter().net().n_in,
                    self.domain().adapter().net().n_y,
                )?
        {
            return Err(invalid("conditional rows lost original full payload claim"));
        }
        Ok(())
    }
    pub(in crate::margin_row) fn lower(&self, requested: Option<Instant>) -> Result<&PassOut> {
        self.checkpoint(requested)?;
        Ok(&self.0.lower)
    }
    pub(in crate::margin_row) fn upper(&self, requested: Option<Instant>) -> Result<&PassOut> {
        self.checkpoint(requested)?;
        Ok(&self.0.upper)
    }
    pub(in crate::margin_row) fn endpoints(
        &self,
        requested: Option<Instant>,
    ) -> Result<(&[f64], &[f64])> {
        self.checkpoint(requested)?;
        Ok((&self.0.lo, &self.0.hi))
    }
    #[expect(dead_code, reason = "prepared collector integration")]
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes() + self.0.frame.resident_bytes()
    }
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

struct EvaluationInner {
    bounds: Vec<f64>,
    ybox: YBox,
    rows: PreparedDomainRows,
    crossing: Option<f64>,
    deadline: Instant,
    receipt: PreparedDomainEvaluationReceipt,
    resident: PersistentReservation,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::margin_row) struct PreparedDomainEvaluationReceipt {
    pub classes: usize,
    pub row_tiles: usize,
    pub margin_tiles: usize,
    pub direct_columns: usize,
    pub difference_residual_terms: usize,
    pub phase_halfspaces: usize,
    pub tightened_heads: usize,
    pub dual_refusals: usize,
}
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedDomainEvaluation(Arc<EvaluationInner>, Instant);
impl PreparedDomainEvaluation {
    pub(in crate::margin_row) fn domain(&self) -> &PreparedDomain {
        self.0.rows.domain()
    }
    pub(in crate::margin_row) fn classes(&self) -> &[usize] {
        self.domain().classes()
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.0
            .rows
            .deadline(Some(requested.map_or(self.1.min(self.0.deadline), |d| {
                d.min(self.1).min(self.0.deadline)
            })))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.rows.checkpoint(Some(at))?;
        let net = self.domain().adapter().net();
        if !self.0.resident.shares_ledger(self.domain().budget())
            || self.0.bounds.len() != self.classes().len()
            || self.0.ybox.ly.len() != net.n_y
            || self.0.ybox.uy.len() != net.n_y
        {
            return Err(invalid(
                "conditional evaluation lost original complete objective",
            ));
        }
        Ok(())
    }
    pub(in crate::margin_row) fn bounds(&self, requested: Option<Instant>) -> Result<&[f64]> {
        self.checkpoint(requested)?;
        Ok(&self.0.bounds)
    }
    pub(in crate::margin_row) fn rows(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedDomainRows> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let mut rows = self.0.rows.clone();
        rows.1 = rows.1.min(at);
        Ok(rows)
    }
    pub(in crate::margin_row) fn ybox(&self, requested: Option<Instant>) -> Result<&YBox> {
        self.checkpoint(requested)?;
        Ok(&self.0.ybox)
    }
    pub(in crate::margin_row) fn crossing(&self) -> Option<f64> {
        self.0.crossing
    }
    pub(in crate::margin_row) fn receipt(&self) -> PreparedDomainEvaluationReceipt {
        self.0.receipt
    }
    #[expect(dead_code, reason = "prepared collector integration")]
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes() + self.0.rows.resident_bytes()
    }
}

pub(in crate::margin_row) fn build_prepared_domain_rows(
    domain: &PreparedDomain,
    requested: Option<Instant>,
) -> Result<PreparedDomainRows> {
    build_rows_inner(domain, requested, None)
}
fn build_rows_inner(
    domain: &PreparedDomain,
    requested: Option<Instant>,
    mut after_tile: Option<&mut dyn FnMut(usize)>,
) -> Result<PreparedDomainRows> {
    let mut at = domain.deadline(requested)?;
    domain.checkpoint(Some(at))?;
    let adapter = domain.adapter();
    let net = adapter.net();
    let budget = domain.budget();
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
    let plan = prepared_ordinary_walk_plan(adapter, at)?;
    census.complete().map_err(refusal)?;
    drop(census);
    let frame = domain.frame(Some(at))?;
    at = at.min(frame.deadline(Some(at))?);
    let n = net.n_in;
    let columns = net.n_y;
    let cells = mul(n, columns)?;
    let mut owner = admit(
        budget,
        WorkRequest {
            persistent_bytes: rows_bytes(n, columns)?,
            work: work(add(mul(cells, 8)?, mul(columns, 64)?)?, 256)?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &owner);
    let mut lower = zero_pass(n, columns, domain, at)?;
    let mut upper = zero_pass(n, columns, domain, at)?;
    let mut tiles = 0;
    let mut first = 0;
    while first < columns {
        domain.cheap_checkpoint(at)?;
        let (mut lease, width) =
            tile_lease(&plan, budget, (columns - first).min(TILE), at, 2, 0, 0)?;
        at = min_lease(at, &lease);
        let mut seed = Seed {
            s: Array2::from_shape_vec((columns, width), filled(mul(columns, width)?, domain, at)?)
                .map_err(|_| invalid("conditional identity shape changed"))?,
            e: None,
        };
        for j in 0..width {
            poll(domain, at, j)?;
            seed.s[[first + j, j]] = 1.0;
        }
        let out = run_prepared_domain_tile(&frame, &seed, LaneDir::Lower, at)?;
        transfer(out, &mut lower, first, width, domain, at)?;
        let out = run_prepared_domain_tile(&frame, &seed, LaneDir::Upper, at)?;
        transfer(out, &mut upper, first, width, domain, at)?;
        drop(seed);
        lease.complete().map_err(refusal)?;
        drop(lease);
        tiles = add(tiles, 1)?;
        if let Some(observer) = after_tile.as_mut() {
            observer(tiles)
        }
        domain.cheap_checkpoint(at)?;
        first = add(first, width)?;
    }
    let lo = concretize_prepared_pass(adapter, &lower, true, at)?;
    let hi = concretize_prepared_pass(adapter, &upper, false, at)?;
    for j in 0..columns {
        poll(domain, at, j)?;
        if !lo[j].is_finite() || !hi[j].is_finite() {
            return Err(invalid("conditional complete endpoints are nonfinite"));
        }
    }
    let actual = add(
        size_of::<RowsInner>() + 2 * size_of::<usize>() + META,
        add(
            pass_bytes(&lower)?,
            add(
                pass_bytes(&upper)?,
                mul(add(lo.capacity(), hi.capacity())?, 8)?,
            )?,
        )?,
    )?;
    if actual > rows_bytes(n, columns)? {
        return Err(invalid("conditional row allocator exceeded preadmission"));
    }
    frame.checkpoint(Some(at))?;
    owner.mark_materialized().map_err(refusal)?;
    let result = PreparedDomainRows(
        Arc::new(RowsInner {
            lower,
            upper,
            lo,
            hi,
            frame,
            plan,
            deadline: at,
            tiles,
            resident: owner.commit_persistent().map_err(refusal)?,
        }),
        at,
    );
    result.checkpoint(Some(at))?;
    Ok(result)
}

pub(in crate::margin_row) fn evaluate_prepared_domain(
    domain: &PreparedDomain,
    requested: Option<Instant>,
) -> Result<PreparedDomainEvaluation> {
    let rows = build_prepared_domain_rows(domain, requested)?;
    evaluate_prepared_domain_with_rows(domain, rows, requested)
}
pub(in crate::margin_row) fn evaluate_prepared_domain_with_rows(
    domain: &PreparedDomain,
    rows: PreparedDomainRows,
    requested: Option<Instant>,
) -> Result<PreparedDomainEvaluation> {
    let mut at = domain.deadline(requested)?.min(rows.deadline(requested)?);
    domain.checkpoint(Some(at))?;
    rows.checkpoint(Some(at))?;
    if !domain.same_domain(rows.domain()) {
        return Err(invalid("conditional evaluator foreign phase row owner"));
    }
    let adapter = domain.adapter();
    let net = adapter.net();
    let budget = domain.budget();
    let ny = net.n_y;
    let nf = domain.classes().len();
    let mut owner = admit(
        budget,
        WorkRequest {
            persistent_bytes: add(
                size_of::<EvaluationInner>() + 2 * size_of::<usize>() + META,
                add(mul(nf, 8)?, mul(ny, 16)?)?,
            )?,
            work: work(add(add(ny, nf)?, mul(ny, add(domain.depth(), 1)?)?)?, 128)?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &owner);
    let mut ybox = YBox {
        ly: copied(&rows.0.lo, domain, at)?,
        uy: copied(&rows.0.hi, domain, at)?,
    };
    for (i, phase) in domain.phases().enumerate() {
        poll(domain, at, i)?;
        if let PreparedPhase::Head { neuron, side } = phase {
            match side {
                PreparedPhaseSide::Active => ybox.ly[neuron] = ybox.ly[neuron].max(0.0),
                PreparedPhaseSide::Inactive => ybox.uy[neuron] = ybox.uy[neuron].min(0.0),
            }
        }
    }
    let mut receipt = PreparedDomainEvaluationReceipt {
        classes: nf,
        row_tiles: rows.0.tiles,
        ..PreparedDomainEvaluationReceipt::default()
    };
    tighten_original_head_aliases(domain, &mut ybox, &mut at, &mut receipt)?;
    tighten_head_region(domain, &rows, &mut ybox, &mut at, &mut receipt)?;
    let mut crossing = None;
    for i in 0..ny {
        poll(domain, at, i)?;
        if ybox.ly[i] > ybox.uy[i] {
            let slack = next_down(ybox.ly[i] - ybox.uy[i]);
            if !slack.is_finite() || slack <= 0.0 {
                return Err(invalid(
                    "conditional crossing lacks finite outward evidence",
                ));
            }
            crossing = Some(crossing.map_or(slack, |s: f64| s.max(slack)));
        }
    }
    let mut bounds = filled(nf, domain, at)?;
    if crossing.is_none() {
        let head = head_gates_request(ny)?;
        let dots = row_dots_request(net.n_in, ny)?;
        let mut helpers = admit(
            budget,
            WorkRequest {
                scratch_bytes: add(head.scratch_bytes, mul(dots.scratch_bytes, 2)?)?,
                work: head
                    .work
                    .checked_add(dots.work.checked_mul(2).ok_or_else(overflow)?)
                    .ok_or_else(overflow)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        at = min_lease(at, &helpers);
        let check = || domain.cheap_checkpoint(at);
        let gates = head_gates_with_checkpoint(
            &ybox,
            RoundMode::Outward,
            adapter.gates().head_alpha.as_deref(),
            &check,
        )?;
        let al_dots = row_dots_with_checkpoint(adapter.gates(), &rows.0.lower, &check)?;
        let au_dots = row_dots_with_checkpoint(adapter.gates(), &rows.0.upper, &check)?;
        let eng = BackwardEngine::new(net, adapter.gates());
        let mut first = 0;
        while first < nf {
            domain.cheap_checkpoint(at)?;
            let mut width = (nf - first).min(TILE);
            let mut lease = loop {
                let walk = match rows.0.plan.request(width) {
                    Ok(q) => q,
                    Err(NyError::CpuMemoryExceeded { .. }) if width > 1 => {
                        width = width.div_ceil(2);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                let mb = margin_batch_request(ny, width)?;
                let seed = margin_seed_request(ny, width, RoundMode::Outward)?;
                let via = compose_viay_request(net.n_in, ny, width)?;
                let direct = direct_from_lower_request(width)?;
                let extra = [mb, seed, via, direct]
                    .iter()
                    .try_fold(0usize, |s, q| add(s, q.scratch_bytes))?;
                let helper_work = [mb, seed, via, direct]
                    .iter()
                    .try_fold(0u64, |s, q| s.checked_add(q.work).ok_or_else(overflow))?;
                let request = WorkRequest {
                    scratch_bytes: add(
                        add(extra, walk.scratch_bytes)?,
                        add(mul(width, 32)?, broadcast_bytes()?)?,
                    )?,
                    work: helper_work
                        .checked_add(walk.work)
                        .and_then(|w| w.checked_add(work(mul(width, ny).ok()?, 128).ok()?))
                        .ok_or_else(overflow)?,
                    ..WorkRequest::default()
                };
                match budget.lease_for(LeaseClass::Fallback, request) {
                    Ok(l) => break l,
                    Err(
                        WorkRefusal::ScratchLimit
                        | WorkRefusal::PersistentLimit
                        | WorkRefusal::MemoryLimit,
                    ) if width > 1 => width = width.div_ceil(2),
                    Err(e) => return Err(refusal(e)),
                }
            };
            at = min_lease(at, &lease);
            let check = || domain.cheap_checkpoint(at);
            let mb = margin_batch_with_checkpoint(
                net,
                domain.target(),
                &domain.classes()[first..first + width],
                &check,
            )?;
            let ms = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check)?;
            let via = compose_viay_with_checkpoint(
                &eng,
                &mb,
                &gates,
                &rows.0.lower,
                &rows.0.upper,
                &al_dots,
                &au_dots,
                RoundMode::Outward,
                &check,
            )?;
            let loss = difference_loss(domain, &mb.adv, &ybox, at, &mut receipt)?;
            let pass = run_prepared_domain_tile(rows.frame(), &ms.seed, LaneDir::Lower, at)?;
            let low = concretize_prepared_pass(adapter, &pass, true, at)?;
            let direct = per_class_direct_from_lower_with_checkpoint(
                &low,
                &ms,
                0..width,
                RoundMode::Outward,
                &check,
            )?;
            receipt.direct_columns = add(receipt.direct_columns, width)?;
            for j in 0..width {
                poll(domain, at, j)?;
                let mut value = ms.m1[j].max(via[j]).max(direct[j]);
                if loss[j] != 0.0 {
                    value = next_down(value - loss[j]);
                }
                if !value.is_finite() {
                    return Err(invalid("conditional original combined margin is nonfinite"));
                }
                bounds[first + j] = value;
            }
            receipt.margin_tiles = add(receipt.margin_tiles, 1)?;
            drop(direct);
            drop(low);
            drop(pass);
            drop(loss);
            drop(via);
            drop(ms);
            drop(mb);
            lease.complete().map_err(refusal)?;
            drop(lease);
            first = add(first, width)?;
        }
        drop(gates);
        drop(al_dots);
        drop(au_dots);
        helpers.complete().map_err(refusal)?;
        drop(helpers);
    }
    domain.checkpoint(Some(at))?;
    rows.checkpoint(Some(at))?;
    owner.mark_materialized().map_err(refusal)?;
    let result = PreparedDomainEvaluation(
        Arc::new(EvaluationInner {
            bounds,
            ybox,
            rows,
            crossing,
            deadline: at,
            receipt,
            resident: owner.commit_persistent().map_err(refusal)?,
        }),
        at,
    );
    result.checkpoint(Some(at))?;
    Ok(result)
}

/// A phase implies a necessary input halfspace built from its OWN complete
/// row error, not a slack calibrated against an intersected scalar interval.
/// Conditional head rows hold on the true trunk region, so their halfspaces
/// safely constrain that same region and every descendant of it.
/// Two byte-identical exact first-head rows consume the SAME actual trunk
/// tensor, so their real preactivations are equal even when upstream bounds
/// are loose. Intersect only genuine explicitly phased coordinates. Row
/// approximation errors or independent folded bias errors forbid this reuse.
/// It is optional: failed work admission preserves the old certified box.
fn tighten_original_head_aliases(
    domain: &PreparedDomain,
    ybox: &mut YBox,
    at: &mut Instant,
    receipt: &mut PreparedDomainEvaluationReceipt,
) -> Result<()> {
    if domain.depth() < 2 {
        return Ok(());
    }
    domain.cheap_checkpoint(*at)?;
    let net = domain.adapter().net();
    let Some(crate::margin_row::net::TwinOp::Gemm {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        shape,
        ..
    }) = net.ops.get(net.i_gemm1)
    else {
        return Err(invalid(
            "conditional original head alias missing source Gemm",
        ));
    };
    if *weight_rel_err != 0.0 {
        return Ok(());
    }
    let cells = mul(mul(domain.depth(), domain.depth())?, add(shape.1, 32)?)?;
    let mut lease = match domain.budget().lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            work: work(add(cells, 32)?, 128)?,
            ..WorkRequest::default()
        },
    ) {
        Ok(lease) => lease,
        Err(_) => {
            // Cancellation/expiry still fails immediately; a finite quota
            // decline grants no work and leaves the previous enclosure intact.
            domain.cheap_checkpoint(*at)?;
            return Ok(());
        }
    };
    *at = min_lease(*at, &lease);
    for (i, left) in domain.phases().enumerate() {
        poll(domain, *at, i)?;
        let PreparedPhase::Head { neuron: a, .. } = left else {
            continue;
        };
        if bias_err[a] != 0.0 {
            continue;
        }
        for (j, right) in domain.phases().enumerate() {
            poll(domain, *at, j)?;
            let PreparedPhase::Head { neuron: b, .. } = right else {
                continue;
            };
            if a >= b || bias_err[b] != 0.0 || bias[a].to_bits() != bias[b].to_bits() {
                continue;
            }
            let mut equal = true;
            for k in 0..shape.1 {
                poll(domain, *at, k)?;
                if weight[a * shape.1 + k].to_bits() != weight[b * shape.1 + k].to_bits() {
                    equal = false;
                    break;
                }
            }
            if equal {
                let lower = ybox.ly[a].max(ybox.ly[b]);
                let upper = ybox.uy[a].min(ybox.uy[b]);
                receipt.tightened_heads = add(
                    receipt.tightened_heads,
                    usize::from(lower > ybox.ly[a] || upper < ybox.uy[a])
                        + usize::from(lower > ybox.ly[b] || upper < ybox.uy[b]),
                )?;
                ybox.ly[a] = lower;
                ybox.ly[b] = lower;
                ybox.uy[a] = upper;
                ybox.uy[b] = upper;
            }
        }
    }
    domain.cheap_checkpoint(*at)?;
    lease.complete().map_err(refusal)?;
    Ok(())
}

fn tighten_head_region(
    domain: &PreparedDomain,
    rows: &PreparedDomainRows,
    ybox: &mut YBox,
    at: &mut Instant,
    receipt: &mut PreparedDomainEvaluationReceipt,
) -> Result<()> {
    let k = domain.depth();
    if k == 0 {
        return Ok(());
    }
    let n = domain.adapter().net().n_in;
    let ny = domain.adapter().net().n_y;
    let quote = dual_lower_outward_request(n, k)?;
    let scratch = add(
        add(mul(k, size_of::<Halfspace>())?, mul(mul(k, n)?, 8)?)?,
        add(mul(n, 8)?, add(quote.scratch_bytes, META)?)?,
    )?;
    let mut lease = admit(
        domain.budget(),
        WorkRequest {
            scratch_bytes: scratch,
            // Every helper callback probes the SAME source ledger and clock.
            // The helper's own 64-unit environment probe bounds callback count
            // by work/64+1; another work+64 conservatively pays our <=64-unit
            // immutable owner checkpoint, independently of row/scope scans.
            work: quote
                .work
                .checked_mul(2)
                .and_then(|v| v.checked_add(64))
                .ok_or_else(overflow)?
                .checked_mul(u64::try_from(mul(ny, 2)?).map_err(|_| overflow())?)
                .and_then(|v| {
                    v.checked_add(
                        work(
                            add(mul(k, n).ok()?, mul(ny, mul(n, 8).ok()?).ok()?).ok()?,
                            128,
                        )
                        .ok()?,
                    )
                })
                .ok_or_else(overflow)?,
            ..WorkRequest::default()
        },
        *at,
    )?;
    *at = min_lease(*at, &lease);
    let mut hs = exact_vec::<Halfspace>(k)?;
    for (i, phase) in domain.phases().enumerate() {
        poll(domain, *at, i)?;
        let halfspace = match phase {
            PreparedPhase::Head { neuron, side } => {
                let pass = if side == PreparedPhaseSide::Inactive {
                    &rows.0.lower
                } else {
                    &rows.0.upper
                };
                let eta = row_eta(domain, pass, neuron, *at)?;
                let mut a = filled(n, domain, *at)?;
                for j in 0..n {
                    poll(domain, *at, j)?;
                    a[j] = if side == PreparedPhaseSide::Inactive {
                        pass.a[[j, neuron]]
                    } else {
                        -pass.a[[j, neuron]]
                    };
                }
                let b = if side == PreparedPhaseSide::Inactive {
                    pass.b[neuron]
                } else {
                    -pass.b[neuron]
                };
                Halfspace {
                    a,
                    d: subtract_eta(b, eta)?,
                }
            }
            PreparedPhase::Trunk {
                layer,
                neuron,
                side,
                ..
            } => {
                let Some(provider) = domain.adapter().clip_rows() else {
                    continue;
                };
                // The selection owner can carry an earlier clock than its
                // shared producer tile. Preserve BOTH before borrowing rows.
                *at = (*at).min(provider.deadline(Some(*at))?);
                let (pair, column) = provider.row(layer, neuron, Some(*at))?;
                *at = (*at).min(pair.deadline());
                let view = if side == PreparedPhaseSide::Inactive {
                    pair.lower(column)
                } else {
                    pair.upper(column)
                }
                .ok_or_else(|| invalid("conditional phase missing original explicit row"))?;
                let eta = view.pointwise_error(*at)?;
                let mut a = filled(n, domain, *at)?;
                for (j, (index, coefficient, _)) in view.entries().enumerate() {
                    poll(domain, *at, j)?;
                    a[index] = if side == PreparedPhaseSide::Inactive {
                        coefficient
                    } else {
                        -coefficient
                    };
                }
                let b = if side == PreparedPhaseSide::Inactive {
                    view.bias()
                } else {
                    -view.bias()
                };
                Halfspace {
                    a,
                    d: subtract_eta(b, eta)?,
                }
            }
        };
        hs.push(halfspace);
    }
    receipt.phase_halfspaces = hs.len();
    let mut obj = filled(n, domain, *at)?;
    for column in 0..ny {
        poll(domain, *at, column)?;
        let lower_eta = row_eta(domain, &rows.0.lower, column, *at)?;
        for j in 0..n {
            poll(domain, *at, j)?;
            obj[j] = rows.0.lower.a[[j, column]];
        }
        let check = || domain.cheap_checkpoint(*at);
        let lower = dual_lower_outward_with_checkpoint(
            &obj,
            &hs,
            &domain.adapter().gates().mid,
            &domain.adapter().gates().rad,
            &check,
        )?;
        let old = (ybox.ly[column], ybox.uy[column]);
        if let Some(bound) = lower {
            let candidate = directed_constant(bound.lower, rows.0.lower.b[column], lower_eta)?;
            ybox.ly[column] = ybox.ly[column].max(candidate);
        } else {
            receipt.dual_refusals = add(receipt.dual_refusals, 1)?;
        }
        let upper_eta = row_eta(domain, &rows.0.upper, column, *at)?;
        for j in 0..n {
            poll(domain, *at, j)?;
            obj[j] = -rows.0.upper.a[[j, column]];
        }
        let upper = dual_lower_outward_with_checkpoint(
            &obj,
            &hs,
            &domain.adapter().gates().mid,
            &domain.adapter().gates().rad,
            &check,
        )?;
        if let Some(bound) = upper {
            let candidate = -directed_constant(bound.lower, -rows.0.upper.b[column], upper_eta)?;
            ybox.uy[column] = ybox.uy[column].min(candidate);
        } else {
            receipt.dual_refusals = add(receipt.dual_refusals, 1)?;
        }
        receipt.tightened_heads = add(
            receipt.tightened_heads,
            usize::from(old.0 < ybox.ly[column] || old.1 > ybox.uy[column]),
        )?;
    }
    rows.checkpoint(Some(*at))?;
    domain.cheap_checkpoint(*at)?;
    drop(obj);
    drop(hs);
    lease.complete().map_err(refusal)?;
    Ok(())
}
fn row_eta(domain: &PreparedDomain, pass: &PassOut, column: usize, at: Instant) -> Result<f64> {
    let errors = pass
        .e
        .as_ref()
        .ok_or_else(|| invalid("conditional row has no error plane"))?;
    let mut eta = pass.eb[column];
    for i in 0..domain.adapter().net().n_in {
        poll(domain, at, i)?;
        let mag = domain.adapter().gates().lo[i]
            .abs()
            .max(domain.adapter().gates().hi[i].abs());
        let e = errors[[i, column]];
        if e != 0.0 && mag != 0.0 {
            eta = next_up(eta + next_up(e * mag));
        }
    }
    if !eta.is_finite() || eta < 0.0 {
        return Err(invalid("conditional row pointwise error overflow"));
    }
    Ok(eta)
}
fn subtract_eta(b: f64, eta: f64) -> Result<f64> {
    let result = if eta == 0.0 { b } else { next_down(b - eta) };
    if !result.is_finite() {
        return Err(invalid("conditional phase constant overflow"));
    }
    Ok(result)
}
fn directed_constant(dot: f64, b: f64, eta: f64) -> Result<f64> {
    let result = next_down(dot + b);
    subtract_eta(result, eta)
}

fn difference_loss(
    domain: &PreparedDomain,
    classes: &[usize],
    ybox: &YBox,
    at: Instant,
    receipt: &mut PreparedDomainEvaluationReceipt,
) -> Result<Vec<f64>> {
    let (weights, bias, (_, ny)) = domain.adapter().net().gemm2();
    let target = domain.target();
    let mut result = filled(classes.len(), domain, at)?;
    for (r, &class) in classes.iter().enumerate() {
        let (_, gap) = two_sum(bias[target], -bias[class]);
        let mut loss = (-gap).max(0.0);
        if gap < 0.0 {
            receipt.difference_residual_terms = add(receipt.difference_residual_terms, 1)?;
        }
        for k in 0..ny {
            poll(domain, at, k)?;
            let (_, gap) = two_sum(weights[target * ny + k], -weights[class * ny + k]);
            if gap < 0.0 {
                loss = next_up(loss + next_up((-gap) * ybox.uy[k].max(0.0)));
                receipt.difference_residual_terms = add(receipt.difference_residual_terms, 1)?;
            }
        }
        if !loss.is_finite() {
            return Err(invalid("conditional original head subtraction overflow"));
        }
        result[r] = loss;
    }
    Ok(result)
}
fn tile_lease(
    plan: &PreparedOrdinaryWalkPlan,
    budget: &WorkBudget,
    desired: usize,
    at: Instant,
    passes: u64,
    extra: usize,
    extra_work: u64,
) -> Result<(WorkLease, usize)> {
    let mut width = desired;
    loop {
        qualify(budget, at)?;
        let mut request = match plan.request(width) {
            Ok(q) => q,
            Err(NyError::CpuMemoryExceeded { .. }) if width > 1 => {
                width = width.div_ceil(2);
                continue;
            }
            Err(e) => return Err(e),
        };
        request.scratch_bytes = add(add(request.scratch_bytes, extra)?, broadcast_bytes()?)?;
        request.work = request
            .work
            .checked_mul(passes)
            .and_then(|v| v.checked_add(extra_work))
            .ok_or_else(overflow)?;
        match budget.lease_for(LeaseClass::Fallback, request) {
            Ok(l) => return Ok((l, width)),
            Err(
                WorkRefusal::ScratchLimit | WorkRefusal::PersistentLimit | WorkRefusal::MemoryLimit,
            ) if width > 1 => width = width.div_ceil(2),
            Err(e) => return Err(refusal(e)),
        }
    }
}
fn zero_pass(n: usize, columns: usize, domain: &PreparedDomain, at: Instant) -> Result<PassOut> {
    let matrix = || {
        Array2::from_shape_vec((n, columns), filled(mul(n, columns)?, domain, at)?)
            .map_err(|_| invalid("conditional plane shape changed"))
    };
    Ok(PassOut {
        a: matrix()?,
        e: Some(matrix()?),
        b: filled(columns, domain, at)?,
        eb: filled(columns, domain, at)?,
        coll: None,
        coll_rows: None,
        unst_rows: None,
    })
}
fn transfer(
    pass: PassOut,
    target: &mut PassOut,
    first: usize,
    width: usize,
    domain: &PreparedDomain,
    at: Instant,
) -> Result<()> {
    let n = domain.adapter().net().n_in;
    let columns = domain.adapter().net().n_y;
    let errors = pass
        .e
        .as_ref()
        .ok_or_else(|| invalid("conditional tile lacks errors"))?;
    if pass.a.dim() != (n, width)
        || errors.dim() != pass.a.dim()
        || pass.b.len() != width
        || pass.eb.len() != width
        || pass.coll.is_some()
        || pass.coll_rows.is_some()
        || pass.unst_rows.is_some()
    {
        return Err(invalid("conditional tile incomplete shape"));
    }
    if !pass.a.is_standard_layout() || !errors.is_standard_layout() {
        return Err(invalid("conditional tile noncontiguous"));
    }
    let PassOut { a, e, b, eb, .. } = pass;
    let (a, a_offset) = a.into_raw_vec_and_offset();
    let (e, e_offset) = e
        .expect("checked actual error plane")
        .into_raw_vec_and_offset();
    let cells = mul(n, width)?;
    if a_offset.unwrap_or(0) != 0
        || e_offset.unwrap_or(0) != 0
        || a.len() != cells
        || e.len() != cells
        || a.capacity() != cells
        || e.capacity() != cells
        || b.capacity() != width
        || eb.capacity() != width
    {
        return Err(invalid(
            "conditional tile capacity exceeds original admission",
        ));
    }
    let dst = target.a.as_slice_mut().expect("actual standard plane");
    let de = target
        .e
        .as_mut()
        .expect("actual error plane")
        .as_slice_mut()
        .expect("actual standard error plane");
    for i in 0..n {
        for j in 0..width {
            poll(domain, at, add(mul(i, width)?, j)?)?;
            let (v, eta) = (a[i * width + j], e[i * width + j]);
            if !v.is_finite() || !eta.is_finite() || eta < 0.0 {
                return Err(invalid("conditional tile invalid actual coefficient/error"));
            }
            dst[i * columns + first + j] = v;
            de[i * columns + first + j] = eta;
        }
    }
    for j in 0..width {
        poll(domain, at, j)?;
        if !b[j].is_finite() || !eb[j].is_finite() || eb[j] < 0.0 {
            return Err(invalid("conditional tile invalid actual constants"));
        }
        target.b[first + j] = b[j];
        target.eb[first + j] = eb[j];
    }
    domain.cheap_checkpoint(at)
}
fn rows_bytes(n: usize, columns: usize) -> Result<usize> {
    add(
        size_of::<RowsInner>() + 2 * size_of::<usize>() + META,
        add(mul(mul(n, columns)?, 32)?, mul(columns, 48)?)?,
    )
}
fn pass_bytes(pass: &PassOut) -> Result<usize> {
    add(
        mul(
            add(pass.a.len(), pass.e.as_ref().map_or(0, |e| e.len()))?,
            8,
        )?,
        mul(add(pass.b.capacity(), pass.eb.capacity())?, 8)?,
    )
}
fn broadcast_bytes() -> Result<usize> {
    add(
        mul(rayon::current_num_threads(), size_of::<Result<()>>())?,
        META,
    )
}
fn copied(values: &[f64], domain: &PreparedDomain, at: Instant) -> Result<Vec<f64>> {
    let mut out = exact_vec(values.len())?;
    for (i, &v) in values.iter().enumerate() {
        poll(domain, at, i)?;
        out.push(v);
    }
    Ok(out)
}
fn filled(n: usize, domain: &PreparedDomain, at: Instant) -> Result<Vec<f64>> {
    let mut out = exact_vec(n)?;
    for i in 0..n {
        poll(domain, at, i)?;
        out.push(0.0);
    }
    Ok(out)
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| invalid("conditional payload allocation refused"))?;
    if out.capacity() != n {
        return Err(invalid("conditional unexpected allocator capacity"));
    }
    Ok(out)
}
fn admit(budget: &WorkBudget, request: WorkRequest, at: Instant) -> Result<WorkLease> {
    qualify(budget, at)?;
    budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(refusal)
}
fn qualify(budget: &WorkBudget, at: Instant) -> Result<()> {
    if Instant::now() >= at {
        return Err(NyError::DeadlineExceeded(
            "prepared conditional CPU evaluation".into(),
        ));
    }
    budget
        .checkpoint_for(LeaseClass::Fallback)
        .map_err(refusal)?;
    ny_core::require_f64_interval_proof_environment()
}
fn poll(domain: &PreparedDomain, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        domain.cheap_checkpoint(at)?;
    }
    Ok(())
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| d.min(at))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn work(n: usize, k: usize) -> Result<u64> {
    u64::try_from(mul(n, k)?).map_err(|_| overflow())
}
fn overflow() -> NyError {
    invalid("conditional request size overflow")
}
fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("prepared domain: {message}"))
}
fn refusal(error: WorkRefusal) -> NyError {
    if error == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("prepared conditional original deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!("prepared conditional resource refusal: {error}"))
    }
}

#[cfg(test)]
#[path = "prepared_domain_eval_tests.rs"]
mod tests;
