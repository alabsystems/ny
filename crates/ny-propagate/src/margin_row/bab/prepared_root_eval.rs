// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual ordinary packs and root evaluation behind retained, checked owners.
//! Weak memo slots never form an adapter -> memo -> adapter ownership cycle.
//! Legacy public Vec/YBox extraction is not the prepared ownership API.

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
use crate::margin_row::engine::{run_prepared_ordinary_tile, LaneDir};
use crate::margin_row::support_program::{
    concretize_prepared_pass, prepared_ordinary_walk_plan, PreparedOrdinaryRowOwner,
    PreparedOrdinaryRows, PreparedOrdinaryWalkPlan, PreparedRootAdapter,
};
use ny_core::dd::two_sum;
use std::mem::size_of;
use std::sync::Weak;

const META: usize = 4096;
const POLL: usize = 256;
const TILE: usize = 64;

/// Last-declared by YPack: actual rows, scalar buffers and dots drop first.
pub(super) struct PreparedPackOwner {
    rows: PreparedOrdinaryRowOwner,
    deadline: Instant,
    resident: PersistentReservation,
}
impl PreparedPackOwner {
    fn adapter(&self) -> &PreparedRootAdapter {
        self.rows.adapter()
    }
    fn budget(&self) -> &WorkBudget {
        self.adapter().source().budget()
    }
    fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.rows.deadline(Some(
            requested.map_or(self.deadline, |at| at.min(self.deadline)),
        ))
    }
    fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.rows.checkpoint(Some(at))?;
        if !self.resident.shares_ledger(self.budget())
            || self.resident.bytes()
                < add(
                    mul(self.adapter().net().n_y, 48)?,
                    add(size_of::<YPack>(), META)?,
                )?
        {
            return Err(invalid("prepared pack lost original scalar/header claim"));
        }
        qualify(self.budget(), at)
    }
    fn matches(&self, eng: &BackwardEngine<'_>) -> bool {
        std::ptr::eq(eng.net, self.adapter().net())
            && std::ptr::eq(eng.root, self.adapter().gates())
            && eng.err_lane() == super::super::errlane::ErrLane::PerCol
            && eng.gpu_conv_route().is_off()
    }
}

struct EvaluationInner {
    target: usize,
    adversaries: Vec<usize>,
    eval: RootEval,
    deadline: Instant,
    receipt: PreparedRootEvaluationReceipt,
    resident: PersistentReservation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::margin_row) struct PreparedRootEvaluationReceipt {
    pub classes: usize,
    pub tiles: usize,
    pub direct_columns: usize,
    pub cached_pack: bool,
    pub shared_payload_copy: bool,
    pub difference_residual_terms: usize,
}

/// Read-only actual RootEval and YPack; Clone shares all original last owners.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedRootEvaluation(Arc<EvaluationInner>, Instant);
impl PreparedRootEvaluation {
    fn pack(&self) -> &Arc<YPack> {
        &self.0.eval.pack
    }
    fn owner(&self) -> &PreparedPackOwner {
        self.pack()
            .prepared_owner
            .as_ref()
            .expect("sealed prepared pack")
    }
    pub(in crate::margin_row) fn adapter(&self) -> &PreparedRootAdapter {
        self.owner().adapter()
    }
    pub(in crate::margin_row) fn target(&self) -> usize {
        self.0.target
    }
    pub(in crate::margin_row) fn adversaries(&self) -> &[usize] {
        &self.0.adversaries
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.owner()
            .deadline(Some(requested.map_or(self.1.min(self.0.deadline), |at| {
                at.min(self.1).min(self.0.deadline)
            })))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.owner().checkpoint(Some(at))?;
        if !self.0.resident.shares_ledger(self.owner().budget())
            || self.0.eval.dj.len() != self.0.adversaries.len()
            || self.0.resident.bytes()
                < add(
                    mul(self.0.adversaries.len(), 24)?,
                    add(size_of::<EvaluationInner>(), META)?,
                )?
            || self.0.receipt.classes != self.0.adversaries.len()
        {
            return Err(invalid(
                "prepared evaluation lost complete original objective",
            ));
        }
        Ok(())
    }
    pub(in crate::margin_row) fn bounds(&self, requested: Option<Instant>) -> Result<&[f64]> {
        self.checkpoint(requested)?;
        Ok(&self.0.eval.dj)
    }
    pub(in crate::margin_row) fn tree_classes(
        &self,
        requested: Option<Instant>,
    ) -> Result<&[usize]> {
        self.checkpoint(requested)?;
        Ok(&self.0.eval.tree_classes)
    }
    pub(in crate::margin_row) fn receipt(&self) -> PreparedRootEvaluationReceipt {
        self.0.receipt
    }
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Only audited parent consumers can borrow the actual legacy structure;
    /// they cannot move its public vectors out of this immutable borrow.
    pub(super) fn legacy_view(&self, requested: Option<Instant>) -> Result<&RootEval> {
        self.checkpoint(requested)?;
        Ok(&self.0.eval)
    }
    pub(in crate::margin_row) fn rows(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedMemoRows> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let mut owner = self.clone();
        owner.1 = owner.1.min(at);
        Ok(PreparedMemoRows { owner })
    }
    pub(in crate::margin_row) fn copy_ybox(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedRootYBox> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let budget = self.owner().budget();
        let n = self.adapter().net().n_y;
        let mut lease = admit(
            budget,
            WorkRequest {
                persistent_bytes: add(mul(n, 16)?, add(size_of::<YBoxInner>(), META)?)?,
                work: work(n, 256)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        at = min_lease(at, &lease);
        let value = YBox {
            ly: copied(&self.pack().ly0, budget, at)?,
            uy: copied(&self.pack().uy0, budget, at)?,
        };
        self.checkpoint(Some(at))?;
        lease.mark_materialized().map_err(refusal)?;
        let result = PreparedRootYBox(Arc::new(YBoxInner {
            value,
            owner: self.clone(),
            deadline: at,
            resident: lease.commit_persistent().map_err(refusal)?,
        }));
        result.view(Some(at))?;
        Ok(result)
    }
}

#[derive(Clone)]
pub(in crate::margin_row) struct PreparedMemoRows {
    owner: PreparedRootEvaluation,
}
impl PreparedMemoRows {
    pub(in crate::margin_row) fn al(&self, requested: Option<Instant>) -> Result<&PassOut> {
        self.owner.checkpoint(requested)?;
        Ok(&self.owner.pack().al)
    }
    pub(in crate::margin_row) fn au(&self, requested: Option<Instant>) -> Result<&PassOut> {
        self.owner.checkpoint(requested)?;
        Ok(&self.owner.pack().au)
    }
    pub(in crate::margin_row) fn evaluation(&self) -> &PreparedRootEvaluation {
        &self.owner
    }
}
struct YBoxInner {
    value: YBox,
    owner: PreparedRootEvaluation,
    deadline: Instant,
    resident: PersistentReservation,
}
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedRootYBox(Arc<YBoxInner>);
impl PreparedRootYBox {
    pub(in crate::margin_row) fn view(&self, requested: Option<Instant>) -> Result<&YBox> {
        let at = self.0.owner.deadline(Some(
            requested.map_or(self.0.deadline, |at| at.min(self.0.deadline)),
        ))?;
        self.0.owner.checkpoint(Some(at))?;
        if !self.0.resident.shares_ledger(self.0.owner.owner().budget())
            || self.0.resident.bytes()
                < add(
                    mul(self.0.value.ly.len(), 16)?,
                    add(size_of::<YBoxInner>(), META)?,
                )?
        {
            return Err(invalid("prepared y-box lost original output claim"));
        }
        Ok(&self.0.value)
    }
}

/// These fixed slots price weak-lived Arc allocations after their strong
/// numerical owners cease. Field order drops weak blocks before byte claims.
struct MemoSlot {
    evaluation: Weak<EvaluationInner>,
    pack: Weak<YPack>,
    #[expect(dead_code, reason = "reservation lives until memo eviction")]
    resident: PersistentReservation,
}
#[derive(Default)]
pub(super) struct PreparedMemoSlots {
    slots: [Option<MemoSlot>; 2],
}
impl PreparedMemoSlots {
    pub(super) fn clear(&mut self) {
        self.slots = [None, None];
    }
}

pub(in crate::margin_row) fn prepared_root_eval_lookup(
    eng: &BackwardEngine<'_>,
    target: usize,
    adversaries: &[usize],
    requested: Option<Instant>,
) -> Result<Option<PreparedRootEvaluation>> {
    // The fixed scan allocates nothing and precedes no unadmitted fingerprint.
    let candidates = {
        let slots = eng
            .root
            .root_eval_memo
            .1
            .try_lock()
            .map_err(memo_lock_error)?;
        if requested.is_some_and(|at| Instant::now() >= at) {
            return Err(NyError::DeadlineExceeded(
                "prepared memo lookup inherited clock".into(),
            ));
        }
        let candidates = [
            slots.slots[0].as_ref().and_then(|s| s.evaluation.upgrade()),
            slots.slots[1].as_ref().and_then(|s| s.evaluation.upgrade()),
        ];
        for candidate in candidates.iter().flatten() {
            let proof = PreparedRootEvaluation(
                candidate.clone(),
                requested.map_or(candidate.deadline, |d| d.min(candidate.deadline)),
            );
            if proof.owner().matches(eng) {
                match proof.checkpoint(requested) {
                    Ok(()) | Err(NyError::DeadlineExceeded(_)) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        candidates
    };
    for candidate in candidates.into_iter().flatten() {
        let inherited = candidate.deadline;
        let mut result =
            PreparedRootEvaluation(candidate, requested.map_or(inherited, |d| d.min(inherited)));
        if !result.owner().matches(eng) {
            continue;
        }
        if result.target() != target {
            continue;
        }
        let mut at = match result.deadline(requested) {
            Ok(at) => at,
            Err(NyError::DeadlineExceeded(_)) => continue,
            Err(error) => return Err(error),
        };
        let mut lease = admit(
            result.owner().budget(),
            WorkRequest {
                work: work(add(adversaries.len(), 1)?, 128)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        at = min_lease(at, &lease);
        result.1 = result.1.min(at);
        let mut same = result.adversaries().len() == adversaries.len();
        if same {
            for (i, (&left, &right)) in result.adversaries().iter().zip(adversaries).enumerate() {
                poll(result.owner().budget(), at, i)?;
                if left != right {
                    same = false;
                    break;
                }
            }
        }
        if same {
            result.checkpoint(Some(at))?;
            lease.complete().map_err(refusal)?;
            return Ok(Some(result));
        }
        lease.complete().map_err(refusal)?;
    }
    Ok(None)
}

fn cached_pack(rows: &PreparedOrdinaryRows, at: Instant) -> Result<Option<Arc<YPack>>> {
    let adapter = rows.adapter();
    let candidates = {
        let slots = adapter
            .gates()
            .root_eval_memo
            .1
            .try_lock()
            .map_err(memo_lock_error)?;
        rows.checkpoint(Some(at))?;
        [
            slots.slots[0].as_ref().and_then(|s| s.pack.upgrade()),
            slots.slots[1].as_ref().and_then(|s| s.pack.upgrade()),
        ]
    };
    for pack in candidates.into_iter().flatten() {
        if let Some(owner) = &pack.prepared_owner {
            if owner.adapter().same_owner(adapter) {
                match owner.checkpoint(Some(at)) {
                    Ok(()) => return Ok(Some(pack)),
                    Err(NyError::DeadlineExceeded(_)) => continue,
                    Err(error) => return Err(error),
                }
            }
        }
    }
    Ok(None)
}

pub(in crate::margin_row) fn build_prepared_root_eval(
    rows: PreparedOrdinaryRows,
    target: usize,
    adversaries: &[usize],
    requested: Option<Instant>,
) -> Result<PreparedRootEvaluation> {
    build_inner(rows, target, adversaries, requested, None)
}

// A test observer may cancel after genuine completed work; it cannot supply
// source facts, row planes or an accepted evaluation.
fn build_inner(
    rows: PreparedOrdinaryRows,
    target: usize,
    adversaries: &[usize],
    requested: Option<Instant>,
    after_tile: Option<&mut dyn FnMut(usize)>,
) -> Result<PreparedRootEvaluation> {
    let mut at = rows.deadline(requested)?;
    rows.checkpoint(Some(at))?;
    let adapter = rows.adapter().clone();
    let budget = adapter.source().budget();
    let net = adapter.net();
    if target >= net.n_out || adversaries.is_empty() || adversaries.len() >= net.n_out {
        return Err(invalid("prepared evaluation invalid target/class count"));
    }
    let words = add(net.n_out / 64, usize::from(!net.n_out.is_multiple_of(64)))?;
    let mut header = admit(
        budget,
        WorkRequest {
            scratch_bytes: add(META, mul(words, size_of::<u64>())?)?,
            work: work(
                add(
                    add(net.ops.len(), net.tsize.len())?,
                    add(add(adversaries.len(), net.n_y)?, words)?,
                )?,
                512,
            )?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &header);
    let mut seen = filled(words, 0u64, budget, at)?;
    for (i, &class) in adversaries.iter().enumerate() {
        poll(budget, at, i)?;
        if class >= net.n_out || class == target {
            return Err(invalid("prepared evaluation out-of-range original class"));
        }
        let bit = 1u64 << (class % 64);
        let word = &mut seen[class / 64];
        if *word & bit != 0 {
            return Err(invalid("prepared evaluation duplicate original class"));
        }
        *word |= bit;
    }
    drop(seen);
    let walk_plan = prepared_ordinary_walk_plan(&adapter, at)?;
    let (pack, cached, shared_copy) = match cached_pack(&rows, at)? {
        Some(pack) => {
            drop(rows);
            (pack, true, false)
        }
        None => {
            let (moved, copied_payload) = rows.with_root_rows(
                Some(at),
                |moved| Ok((moved, false)),
                |rows| Ok((rows.copy_into_root_rows(Some(at))?, true)),
            )?;
            let parts = moved.into_parts();
            // Bind the guard first so error unwinding drops numerical vectors
            // before their original claim, just as the published YPack does.
            let row_owner = parts.4;
            let (al, au, ly0, uy0) = (parts.0, parts.1, parts.2, parts.3);
            let quote = row_dots_request(net.n_in, net.n_y)?;
            let mut storage = admit(
                budget,
                WorkRequest {
                    persistent_bytes: add(mul(net.n_y, 48)?, add(size_of::<YPack>(), META)?)?,
                    scratch_bytes: add(quote.scratch_bytes, quote.scratch_bytes)?,
                    work: quote
                        .work
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(work(net.n_y, 64).ok()?))
                        .ok_or_else(overflow)?,
                    ..WorkRequest::default()
                },
                at,
            )?;
            at = min_lease(at, &storage);
            let check = || qualify(budget, at);
            let al_dots = row_dots_with_checkpoint(adapter.gates(), &al, &check)?;
            let au_dots = row_dots_with_checkpoint(adapter.gates(), &au, &check)?;
            let ly_fwd = copied(&ly0, budget, at)?;
            let uy_fwd = copied(&uy0, budget, at)?;
            row_owner.checkpoint(Some(at))?;
            storage.mark_materialized().map_err(refusal)?;
            let resident = storage.commit_persistent().map_err(refusal)?;
            let pack = Arc::new(YPack {
                al,
                au,
                ly0,
                uy0,
                ly_fwd,
                uy_fwd,
                head_slack: None,
                head_group_source: None,
                al_dots,
                au_dots,
                prepared_owner: Some(PreparedPackOwner {
                    rows: row_owner,
                    deadline: at,
                    resident,
                }),
            });
            (pack, false, copied_payload)
        }
    };
    header.complete().map_err(refusal)?;
    drop(header);
    let result = evaluate_pack(
        pack,
        target,
        adversaries,
        at,
        cached,
        shared_copy,
        walk_plan,
        after_tile,
    )?;
    store(&result)?;
    Ok(result)
}

/// Actual per-root memo ingress. A hit reuses the owned canonical RootEval;
/// a miss consumes genuine rows or admits an explicit shared-payload copy.
pub(in crate::margin_row) fn prepared_root_eval_reuse(
    rows: PreparedOrdinaryRows,
    target: usize,
    adversaries: &[usize],
    requested: Option<Instant>,
) -> Result<PreparedRootEvaluation> {
    let at = rows.deadline(requested)?;
    rows.checkpoint(Some(at))?;
    let eng = BackwardEngine::new(rows.adapter().net(), rows.adapter().gates());
    if let Some(result) = prepared_root_eval_lookup(&eng, target, adversaries, Some(at))? {
        return Ok(result);
    }
    build_prepared_root_eval(rows, target, adversaries, Some(at))
}

fn memo_lock_error<T>(error: std::sync::TryLockError<T>) -> NyError {
    match error {
        std::sync::TryLockError::WouldBlock => refusal(WorkRefusal::Unavailable),
        std::sync::TryLockError::Poisoned(_) => invalid("prepared memo poisoned"),
    }
}

fn store(result: &PreparedRootEvaluation) -> Result<()> {
    store_inner(result, None)
}

// The observer can cancel only after an actual two-slot transaction was
// installed. Complete old slot/claim state is retained until its last check.
fn store_inner(
    result: &PreparedRootEvaluation,
    mut after_insert: Option<&mut dyn FnMut()>,
) -> Result<()> {
    let mut at = result.deadline(None)?;
    let mut lease = admit(
        result.owner().budget(),
        WorkRequest {
            persistent_bytes: add(add(size_of::<EvaluationInner>(), size_of::<YPack>())?, META)?,
            work: 256,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &lease);
    result.checkpoint(Some(at))?;
    lease.mark_materialized().map_err(refusal)?;
    let slot = MemoSlot {
        evaluation: Arc::downgrade(&result.0),
        pack: Arc::downgrade(result.pack()),
        resident: lease.commit_persistent().map_err(refusal)?,
    };
    let mut guard = result
        .adapter()
        .gates()
        .root_eval_memo
        .1
        .try_lock()
        .map_err(memo_lock_error)?;
    result.checkpoint(Some(at))?;
    let mut old = std::mem::take(&mut guard.slots);
    guard.slots = [Some(slot), old[0].take()];
    if let Some(observer) = after_insert.as_mut() {
        observer();
    }
    if let Err(error) = result.checkpoint(Some(at)) {
        let failed = guard.slots[0].take();
        guard.slots = [guard.slots[1].take(), old[1].take()];
        drop(failed);
        return Err(error);
    }
    // Final infallible commit. Readers recheck this owner when consuming it;
    // no late failure after this point can displace the old cache on an Err.
    drop(old);
    Ok(())
}

fn evaluate_pack(
    pack: Arc<YPack>,
    target: usize,
    adversaries: &[usize],
    mut at: Instant,
    cached: bool,
    shared_copy: bool,
    walk_plan: PreparedOrdinaryWalkPlan,
    mut after_tile: Option<&mut dyn FnMut(usize)>,
) -> Result<PreparedRootEvaluation> {
    let owner = pack
        .prepared_owner
        .as_ref()
        .ok_or_else(|| invalid("unowned prepared pack"))?;
    at = owner.deadline(Some(at))?;
    let adapter = owner.adapter();
    let budget = owner.budget();
    let net = adapter.net();
    let n_y = net.n_y;
    let mut persistent = admit(
        budget,
        WorkRequest {
            persistent_bytes: add(
                mul(adversaries.len(), 24)?,
                add(size_of::<EvaluationInner>(), META)?,
            )?,
            work: work(add(adversaries.len(), n_y)?, 256)?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &persistent);
    let fixed = head_gates_request(n_y)?;
    let mut heads = admit(
        budget,
        WorkRequest {
            scratch_bytes: add(fixed.scratch_bytes, mul(n_y, 16)?)?,
            work: fixed.work,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &heads);
    let ybox = YBox {
        ly: copied(&pack.ly0, budget, at)?,
        uy: copied(&pack.uy0, budget, at)?,
    };
    let check = || qualify(budget, at);
    let gates = head_gates_with_checkpoint(
        &ybox,
        RoundMode::Outward,
        adapter.gates().head_alpha.as_deref(),
        &check,
    )?;
    let classes = copied(adversaries, budget, at)?;
    let mut dj = filled(adversaries.len(), 0.0, budget, at)?;
    let mut pending = exact_vec::<usize>(adversaries.len(), budget, at)?;
    let mut receipt = PreparedRootEvaluationReceipt {
        classes: adversaries.len(),
        cached_pack: cached,
        shared_payload_copy: shared_copy,
        ..PreparedRootEvaluationReceipt::default()
    };
    let eng = BackwardEngine::new(net, adapter.gates());
    let mut first = 0;
    while first < adversaries.len() {
        qualify(budget, at)?;
        let mut width = (adversaries.len() - first).min(TILE);
        let mut tile = loop {
            let walk = match walk_plan.request(width) {
                Ok(request) => request,
                Err(NyError::CpuMemoryExceeded { .. }) if width > 1 => {
                    width = width.div_ceil(2);
                    continue;
                }
                Err(error) => return Err(error),
            };
            let mb = margin_batch_request(n_y, width)?;
            let seed = margin_seed_request(n_y, width, RoundMode::Outward)?;
            let via = compose_viay_request(net.n_in, n_y, width)?;
            let direct = direct_from_lower_request(width)?;
            let helper_bytes = [mb, seed, via, direct]
                .iter()
                .try_fold(0usize, |sum, q| add(sum, q.scratch_bytes))?;
            let helper_work = [mb, seed, via, direct]
                .iter()
                .try_fold(0u64, |sum, q| sum.checked_add(q.work).ok_or_else(overflow))?;
            let request = WorkRequest {
                scratch_bytes: add(
                    add(helper_bytes, walk.scratch_bytes)?,
                    add(mul(width, 32)?, META)?,
                )?,
                work: helper_work
                    .checked_add(walk.work)
                    .and_then(|n| {
                        n.checked_add(work(mul(width, add(n_y, 1).ok()?).ok()?, 256).ok()?)
                    })
                    .ok_or_else(overflow)?,
                ..WorkRequest::default()
            };
            match budget.lease_for(LeaseClass::Fallback, request) {
                Ok(lease) => break lease,
                Err(
                    WorkRefusal::ScratchLimit
                    | WorkRefusal::PersistentLimit
                    | WorkRefusal::MemoryLimit,
                ) if width > 1 => width = width.div_ceil(2),
                Err(error) => return Err(refusal(error)),
            }
        };
        at = min_lease(at, &tile);
        let check = || qualify(budget, at);
        let mb =
            margin_batch_with_checkpoint(net, target, &adversaries[first..first + width], &check)?;
        let ms = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check)?;
        let via = compose_viay_with_checkpoint(
            &eng,
            &mb,
            &gates,
            &pack.al,
            &pack.au,
            &pack.al_dots,
            &pack.au_dots,
            RoundMode::Outward,
            &check,
        )?;
        // Both independently certified component bounds retain the same
        // nominal head functional; original difference loss is subtracted.
        let loss = difference_loss(net, target, &mb.adv, &ybox, budget, at, &mut receipt)?;
        let need_direct = (0..width).any(|j| {
            !root_schedules_closed(RoundMode::Outward, corrected(ms.m1[j].max(via[j]), loss[j]))
        });
        let direct = if need_direct {
            let pass = run_prepared_ordinary_tile(adapter, &ms.seed, LaneDir::Lower, at)?;
            let low = concretize_prepared_pass(adapter, &pass, true, at)?;
            let bounds = per_class_direct_from_lower_with_checkpoint(
                &low,
                &ms,
                0..width,
                RoundMode::Outward,
                &check,
            )?;
            receipt.direct_columns = add(receipt.direct_columns, width)?;
            Some(bounds)
        } else {
            None
        };
        for j in 0..width {
            poll(budget, at, j)?;
            let mut value = ms.m1[j].max(via[j]);
            if let Some(direct) = &direct {
                value = value.max(direct[j]);
            }
            value = corrected(value, loss[j]);
            require_finite("prepared combined root bound", &[value])?;
            dj[first + j] = value;
            if !root_schedules_closed(RoundMode::Outward, value) {
                pending.push(adversaries[first + j]);
            }
        }
        receipt.tiles = add(receipt.tiles, 1)?;
        if let Some(observer) = after_tile.as_mut() {
            observer(receipt.tiles);
        }
        qualify(budget, at)?;
        // Scratch remains admitted through every simultaneous helper output.
        drop(direct);
        drop(loss);
        drop(via);
        drop(ms);
        drop(mb);
        tile.complete().map_err(refusal)?;
        drop(tile);
        first = add(first, width)?;
    }
    drop(gates);
    drop(ybox);
    heads.complete().map_err(refusal)?;
    drop(heads);
    owner.checkpoint(Some(at))?;
    persistent.mark_materialized().map_err(refusal)?;
    let resident = persistent.commit_persistent().map_err(refusal)?;
    let result = PreparedRootEvaluation(
        Arc::new(EvaluationInner {
            target,
            adversaries: classes,
            eval: RootEval {
                dj,
                tree_classes: pending,
                pack,
                spec_alpha: None,
            },
            deadline: at,
            receipt,
            resident,
        }),
        at,
    );
    result.checkpoint(Some(at))?;
    Ok(result)
}

fn difference_loss(
    net: &TwinNet,
    target: usize,
    adversaries: &[usize],
    ybox: &YBox,
    budget: &WorkBudget,
    at: Instant,
    receipt: &mut PreparedRootEvaluationReceipt,
) -> Result<Vec<f64>> {
    let (w, b, (_, ny)) = net.gemm2();
    let mut result = filled(adversaries.len(), 0.0, budget, at)?;
    for (r, &j) in adversaries.iter().enumerate() {
        let (_, db) = two_sum(b[target], -b[j]);
        let mut loss = (-db).max(0.0);
        if db < 0.0 {
            receipt.difference_residual_terms = add(receipt.difference_residual_terms, 1)?;
        }
        for k in 0..ny {
            poll(budget, at, r * ny + k)?;
            let (_, gap) = two_sum(w[target * ny + k], -w[j * ny + k]);
            if gap < 0.0 {
                let term = next_up((-gap) * ybox.uy[k].max(0.0));
                loss = next_up(loss + term);
                receipt.difference_residual_terms = add(receipt.difference_residual_terms, 1)?;
            }
        }
        require_finite("prepared original difference loss", &[loss])?;
        result[r] = loss;
    }
    Ok(result)
}
fn corrected(value: f64, loss: f64) -> f64 {
    if loss == 0.0 {
        value
    } else {
        next_down(value - loss)
    }
}
fn qualify(budget: &WorkBudget, at: Instant) -> Result<()> {
    if Instant::now() >= at {
        return Err(NyError::DeadlineExceeded("prepared root evaluation".into()));
    }
    budget
        .checkpoint_for(LeaseClass::Fallback)
        .map_err(refusal)?;
    ny_core::require_f64_interval_proof_environment()
}
fn admit(budget: &WorkBudget, request: WorkRequest, at: Instant) -> Result<WorkLease> {
    qualify(budget, at)?;
    let lease = budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(refusal)?;
    qualify(budget, at)?;
    Ok(lease)
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| d.min(at))
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        qualify(budget, at)?;
    }
    Ok(())
}
fn exact_vec<T>(n: usize, budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    qualify(budget, at)?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| invalid("prepared output allocation failed"))?;
    qualify(budget, at)?;
    if out.capacity() != n {
        return Err(invalid("prepared output actual capacity exceeds admission"));
    }
    Ok(out)
}
fn filled<T: Copy>(n: usize, value: T, budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    let mut out = exact_vec(n, budget, at)?;
    for i in 0..n {
        poll(budget, at, i)?;
        out.push(value);
    }
    qualify(budget, at)?;
    Ok(out)
}
fn copied<T: Copy>(source: &[T], budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    let mut out = exact_vec(source.len(), budget, at)?;
    for (i, &v) in source.iter().enumerate() {
        poll(budget, at, i)?;
        out.push(v);
    }
    qualify(budget, at)?;
    Ok(out)
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
fn invalid(msg: &'static str) -> NyError {
    NyError::InvalidSpec(format!("margin_row: {msg}"))
}
fn overflow() -> NyError {
    invalid("prepared root evaluation resource overflow")
}
fn refusal(error: WorkRefusal) -> NyError {
    if error == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("prepared root evaluation resource deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!(
            "prepared root evaluation resource refusal: {error}"
        ))
    }
}

#[cfg(test)]
#[path = "prepared_root_eval_tests.rs"]
mod tests;
