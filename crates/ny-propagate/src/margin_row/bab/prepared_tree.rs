// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Paid best-first enumeration of genuine original phase regions. The old
//! driver remains the complete fallback for unsupported runtime products.
//! A prepared leaf keeps its conditional evaluator, complete class bounds
//! and authenticated path; scalar scheduling advice never grants closure.

use super::prepared_domain_eval::{
    build_prepared_domain_rows, evaluate_prepared_domain_with_rows, PreparedDomainEvaluation,
    PreparedDomainRows,
};
use crate::certify::margin_row as cert_mr;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkRefusal, WorkRequest,
};
use crate::certify::{PaidCoverError, PaidCoverLedger, PaidTreeCover, TreeCover};
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::support_program::{
    PreparedDomain, PreparedDomainPath, PreparedPhaseSide, PreparedRootOpen,
};
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::{cmp::Ordering, collections::BinaryHeap, fmt, mem::size_of, time::Instant};

const META: usize = 4096;
const POLL: usize = 256;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedTreeSchedule {
    Serial,
    Parallel { width: usize },
    Wave { width: usize },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedTreeOptions {
    pub max_expansions: usize,
    pub lru_capacity: usize,
    pub schedule: PreparedTreeSchedule,
}
impl Default for PreparedTreeOptions {
    fn default() -> Self {
        Self {
            max_expansions: 1000,
            lru_capacity: 4,
            schedule: PreparedTreeSchedule::Serial,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparedTreeReceipt {
    pub expansions: usize,
    pub evaluated_domains: usize,
    pub bound_leaves: usize,
    pub empty_leaves: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub parallel_batches: usize,
    pub wave_batches: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedTreeOpenReason {
    ExpansionLimit,
    NoSplit,
    ResourceRefusal,
}

pub enum PreparedTreeOutcome {
    Covered(PreparedTreeCertificate),
    Open(PreparedTreeOpen),
}
pub struct PreparedTreeCertificate {
    state: PreparedTreeState,
    cover: PaidTreeCover,
}
pub struct PreparedTreeOpen {
    state: PreparedTreeState,
    reason: PreparedTreeOpenReason,
}

struct PreparedDomainEntry {
    bound: f64,
    seq: usize,
    evaluation: PreparedDomainEvaluation,
}
impl PartialEq for PreparedDomainEntry {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}
impl Eq for PreparedDomainEntry {}
impl PartialOrd for PreparedDomainEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PreparedDomainEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .bound
            .total_cmp(&self.bound)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
struct CacheEntry {
    rows: PreparedDomainRows,
    tick: usize,
}
struct PreparedLru {
    entries: Vec<CacheEntry>,
    capacity: usize,
    tick: usize,
}
/// Every payload is a producer-authenticated pair for this exact domain.
struct PreparedEvalSeeded {
    domain: PreparedDomain,
    rows: PreparedDomainRows,
}
struct ClosedLeaf {
    evaluation: PreparedDomainEvaluation,
    path: PreparedDomainPath,
}
struct PreparedTreeState {
    root: PreparedDomain,
    frontier: BinaryHeap<PreparedDomainEntry>,
    closed: Vec<ClosedLeaf>,
    lru: PreparedLru,
    parents: Vec<PreparedDomainEntry>,
    children: Vec<PreparedEvalSeeded>,
    results: Vec<Option<Result<PreparedDomainEvaluation>>>,
    classes_bitmap: Vec<u64>,
    ledger: PaidCoverLedger,
    options: PreparedTreeOptions,
    receipt: PreparedTreeReceipt,
    seq: usize,
    root_pending: bool,
    deadline: Instant,
    // All container allocations drop before their single original claim.
    resident: PersistentReservation,
}
fn invalid(text: &str) -> NyError {
    NyError::InvalidSpec(format!("prepared tree: {text}"))
}
fn refusal(e: WorkRefusal) -> NyError {
    if e == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("prepared tree inherited resource deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!("prepared tree resource refusal: {e}"))
    }
}
fn cover_error(e: PaidCoverError) -> NyError {
    match e {
        PaidCoverError::Resource(r) => refusal(r),
        e => invalid(&e.to_string()),
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| refusal(WorkRefusal::Overflow))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| refusal(WorkRefusal::Overflow))
}
fn work(n: usize, factor: u64) -> Result<u64> {
    u64::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(factor))
        .ok_or_else(|| refusal(WorkRefusal::Overflow))
}
fn vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| refusal(WorkRefusal::MemoryLimit))?;
    if v.capacity() > capacity {
        return Err(refusal(WorkRefusal::MemoryLimit));
    }
    Ok(v)
}
fn poll(domain: &PreparedDomain, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        domain.cheap_checkpoint(at)?;
    }
    Ok(())
}
fn recoverable(e: &NyError) -> bool {
    matches!(e, NyError::UnsupportedConfiguration(_))
}
fn width(options: PreparedTreeOptions) -> Result<usize> {
    let n = match options.schedule {
        PreparedTreeSchedule::Serial => 1,
        PreparedTreeSchedule::Parallel { width } | PreparedTreeSchedule::Wave { width } => width,
    };
    if n == 0 {
        return Err(invalid("schedule width must be positive"));
    }
    Ok(n.min(add(options.max_expansions, 1)?))
}

impl PreparedTreeState {
    fn new(
        open: PreparedRootOpen,
        options: PreparedTreeOptions,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let root = PreparedDomain::from_open(open, requested)?;
        let mut at = root.deadline(requested)?;
        root.checkpoint(Some(at))?;
        let capacity = add(options.max_expansions, 1)?;
        let width = width(options)?;
        let children = mul(width, 2)?;
        let bitmap = add(root.adapter().net().gemm2().2 .0, 63)? / 64;
        let tree_entries = mul(
            capacity,
            add(size_of::<PreparedDomainEntry>(), size_of::<ClosedLeaf>())?,
        )?;
        let cache_bytes = mul(options.lru_capacity, size_of::<CacheEntry>())?;
        let parent_bytes = mul(width, size_of::<PreparedDomainEntry>())?;
        let child_bytes = mul(
            children,
            add(
                size_of::<PreparedEvalSeeded>(),
                size_of::<Option<Result<PreparedDomainEvaluation>>>(),
            )?,
        )?;
        let bytes = [
            META,
            size_of::<Self>(),
            tree_entries,
            cache_bytes,
            parent_bytes,
            child_bytes,
            mul(bitmap, size_of::<u64>())?,
        ]
        .into_iter()
        .try_fold(0usize, add)?;
        let budget = root.budget();
        let mut lease = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: bytes,
                    work: work(add(bitmap, add(capacity, options.lru_capacity)?)?, 8)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        let frontier = BinaryHeap::from(vec(capacity)?);
        let closed = vec(capacity)?;
        let entries = vec(options.lru_capacity)?;
        let parents = vec(width)?;
        let child_buffer = vec(children)?;
        let results = vec(children)?;
        let mut classes_bitmap = vec(bitmap)?;
        for i in 0..bitmap {
            poll(&root, at, i)?;
            classes_bitmap.push(0);
        }
        let ledger = PaidCoverLedger::new(budget, capacity, at).map_err(cover_error)?;
        root.cheap_checkpoint(at)?;
        lease.mark_materialized().map_err(refusal)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        let state = Self {
            root,
            frontier,
            closed,
            lru: PreparedLru {
                entries,
                capacity: options.lru_capacity,
                tick: 0,
            },
            parents,
            children: child_buffer,
            results,
            classes_bitmap,
            ledger,
            options,
            receipt: PreparedTreeReceipt::default(),
            seq: 0,
            root_pending: true,
            deadline: at,
            resident,
        };
        state.checkpoint(None)?;
        Ok(state)
    }
    fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = requested.map_or(self.deadline, |d| d.min(self.deadline));
        self.root.checkpoint(Some(at))?;
        if !self.resident.shares_ledger(self.root.budget()) {
            return Err(invalid("foreign container claim"));
        }
        self.ledger.checkpoint(Some(at)).map_err(cover_error)
    }
    fn control_lease(&self, units: usize) -> Result<crate::certify::work::WorkLease> {
        self.root.cheap_checkpoint(self.deadline)?;
        self.root
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: work(units, 128)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)
    }
    fn rows_for(&mut self, domain: &PreparedDomain) -> Result<PreparedDomainRows> {
        let mut lease = self.control_lease(add(self.lru.entries.len(), 1)?)?;
        self.lru.tick = add(self.lru.tick, 1)?;
        for (i, entry) in self.lru.entries.iter_mut().enumerate() {
            poll(domain, self.deadline, i)?;
            if entry.rows.domain().same_domain(domain) {
                entry.rows.checkpoint(Some(self.deadline))?;
                entry.tick = self.lru.tick;
                self.receipt.cache_hits = add(self.receipt.cache_hits, 1)?;
                lease.complete().map_err(refusal)?;
                return Ok(entry.rows.clone());
            }
        }
        self.receipt.cache_misses = add(self.receipt.cache_misses, 1)?;
        lease.complete().map_err(refusal)?;
        drop(lease);
        let rows = build_prepared_domain_rows(domain, Some(self.deadline))?;
        self.deadline = self.deadline.min(rows.deadline(None)?);
        self.cache_rows(&rows)?;
        Ok(rows)
    }
    fn cache_rows(&mut self, rows: &PreparedDomainRows) -> Result<()> {
        if self.lru.capacity == 0 {
            return Ok(());
        }
        let mut lease = self.control_lease(add(self.lru.entries.len(), 1)?)?;
        rows.checkpoint(Some(self.deadline))?;
        if !rows.domain().same_root(&self.root) {
            return Err(invalid("cache rows belong to foreign original root"));
        }
        let mut oldest = 0usize;
        for (i, entry) in self.lru.entries.iter().enumerate() {
            poll(&self.root, self.deadline, i)?;
            if entry.rows.domain().same_domain(rows.domain()) {
                lease.complete().map_err(refusal)?;
                return Ok(());
            }
            if entry.tick < self.lru.entries[oldest].tick {
                oldest = i;
            }
        }
        self.lru.tick = add(self.lru.tick, 1)?;
        let entry = CacheEntry {
            rows: rows.clone(),
            tick: self.lru.tick,
        };
        if self.lru.entries.len() < self.lru.capacity {
            self.lru.entries.push(entry)
        } else {
            self.lru.entries[oldest] = entry;
        }
        lease.complete().map_err(refusal)?;
        Ok(())
    }
    fn eval_prepare(&mut self, domain: PreparedDomain) -> Result<PreparedEvalSeeded> {
        let rows = self.rows_for(&domain)?;
        if !rows.domain().same_domain(&domain) {
            return Err(invalid("conditional seed rows relabelled"));
        }
        Ok(PreparedEvalSeeded { domain, rows })
    }
    fn eval_main(seeded: &PreparedEvalSeeded, at: Instant) -> Result<PreparedDomainEvaluation> {
        evaluate_prepared_domain_with_rows(&seeded.domain, seeded.rows.clone(), Some(at))
    }
    fn classify(&mut self, evaluation: &PreparedDomainEvaluation) -> Result<(bool, f64)> {
        self.deadline = self.deadline.min(evaluation.deadline(None)?);
        evaluation.checkpoint(Some(self.deadline))?;
        if !evaluation.domain().same_root(&self.root) {
            return Err(invalid("foreign leaf root"));
        }
        let classes = evaluation.classes();
        let bounds = evaluation.bounds(Some(self.deadline))?;
        let nout = self.root.adapter().net().gemm2().2 .0;
        let mut lease =
            self.control_lease(add(self.classes_bitmap.len(), mul(classes.len(), 4)?)?)?;
        if classes.len() != nout.saturating_sub(1) || bounds.len() != classes.len() {
            return Err(invalid("incomplete original class coverage"));
        }
        for (i, v) in self.classes_bitmap.iter_mut().enumerate() {
            poll(&self.root, self.deadline, i)?;
            *v = 0;
        }
        let mut minimum = f64::INFINITY;
        let mut closed = true;
        for (i, ((&actual, &expected), &bound)) in classes
            .iter()
            .zip(self.root.classes())
            .zip(bounds)
            .enumerate()
        {
            poll(&self.root, self.deadline, i)?;
            if actual != expected || actual >= nout || actual == self.root.target() {
                return Err(invalid("changed original class order or target"));
            }
            let mask = 1u64 << (actual % 64);
            let word = &mut self.classes_bitmap[actual / 64];
            if *word & mask != 0 {
                return Err(invalid("duplicate class leaf"));
            }
            *word |= mask;
            if !bound.is_finite() {
                return Err(invalid("nonfinite bound leaf"));
            }
            minimum = minimum.min(bound);
            closed &= bound > 0.0;
        }
        lease.complete().map_err(refusal)?;
        if let Some(slack) = evaluation.crossing() {
            if !slack.is_finite() || slack <= 0.0 {
                return Err(invalid("invalid conditional emptiness slack"));
            }
            return Ok((true, slack));
        }
        Ok((closed, minimum))
    }
    fn eval_finish(&mut self, evaluation: PreparedDomainEvaluation) -> Result<()> {
        let (closed, minimum) = self.classify(&evaluation)?;
        self.deadline = self.deadline.min(evaluation.deadline(None)?);
        if closed {
            let path = evaluation.domain().cover_path(Some(self.deadline))?;
            if !path.domain().same_domain(evaluation.domain()) {
                return Err(invalid("leaf path region mismatch"));
            }
            let evidence = if evaluation.crossing().is_some() {
                cert_mr::empty_evidence(RoundMode::Outward, minimum)
            } else {
                cert_mr::bound_evidence(RoundMode::Outward, minimum)
            };
            self.ledger
                .leaf(
                    Some(path.path(Some(self.deadline))?),
                    evidence,
                    Some(self.deadline),
                )
                .map_err(cover_error)?;
            if evaluation.crossing().is_some() {
                self.receipt.empty_leaves = add(self.receipt.empty_leaves, 1)?
            } else {
                self.receipt.bound_leaves = add(self.receipt.bound_leaves, 1)?
            }
            self.closed.push(ClosedLeaf { evaluation, path });
        } else {
            let seq = self.seq;
            self.seq = add(seq, 1)?;
            self.frontier.push(PreparedDomainEntry {
                bound: minimum,
                seq,
                evaluation,
            });
        }
        self.receipt.evaluated_domains = add(self.receipt.evaluated_domains, 1)?;
        Ok(())
    }
    fn root_entry(&mut self) -> Result<()> {
        let root = self.root.clone();
        let seed = self.eval_prepare(root)?;
        let evaluation = Self::eval_main(&seed, self.deadline)?;
        self.eval_finish(evaluation)?;
        self.root_pending = false;
        Ok(())
    }
    fn restore_parents(&mut self) {
        while let Some(parent) = self.parents.pop() {
            self.frontier.push(parent);
        }
        self.children.clear();
        self.results.clear();
    }
    fn schedule_batch(&mut self) -> Result<Option<PreparedTreeOpenReason>> {
        self.schedule_batch_inner(None)
    }
    fn schedule_batch_inner(
        &mut self,
        mut after_leaf: Option<&mut dyn FnMut(&mut Self, usize)>,
    ) -> Result<Option<PreparedTreeOpenReason>> {
        let remaining = self.options.max_expansions - self.receipt.expansions;
        if remaining == 0 {
            return Ok(Some(PreparedTreeOpenReason::ExpansionLimit));
        }
        let width = width(self.options)?.min(remaining);
        let depth = self.frontier.peek().map(|e| e.evaluation.domain().depth());
        let mut control = self.control_lease(add(mul(width, 8)?, self.frontier.len())?)?;
        while self.parents.len() < width {
            self.root.cheap_checkpoint(self.deadline)?;
            if matches!(self.options.schedule, PreparedTreeSchedule::Wave { .. })
                && self
                    .frontier
                    .peek()
                    .is_some_and(|e| Some(e.evaluation.domain().depth()) != depth)
            {
                break;
            }
            let Some(parent) = self.frontier.pop() else {
                break;
            };
            self.parents.push(parent);
        }
        control.complete().map_err(refusal)?;
        drop(control);
        for i in 0..self.parents.len() {
            let domain = self.parents[i].evaluation.domain().clone();
            let Some(site) = domain.next_split(Some(self.deadline))? else {
                self.restore_parents();
                return Ok(Some(PreparedTreeOpenReason::NoSplit));
            };
            for side in [PreparedPhaseSide::Inactive, PreparedPhaseSide::Active] {
                let child = domain.split(site, side, Some(self.deadline))?;
                let seed = self.eval_prepare(child)?;
                self.children.push(seed);
            }
        }
        for _ in 0..self.children.len() {
            self.results.push(None);
        }
        match self.options.schedule {
            PreparedTreeSchedule::Serial => {
                for (slot, seed) in self.results.iter_mut().zip(&self.children) {
                    *slot = Some(Self::eval_main(seed, self.deadline));
                }
            }
            PreparedTreeSchedule::Parallel { .. } | PreparedTreeSchedule::Wave { .. } => {
                self.results
                    .par_iter_mut()
                    .zip(self.children.par_iter())
                    .for_each(|(slot, seed)| {
                        *slot = Some(Self::eval_main(seed, self.deadline));
                    });
                if matches!(self.options.schedule, PreparedTreeSchedule::Wave { .. }) {
                    self.receipt.wave_batches = add(self.receipt.wave_batches, 1)?
                } else {
                    self.receipt.parallel_batches = add(self.receipt.parallel_batches, 1)?
                }
            }
        }
        for result in &mut self.results {
            if let Some(Err(e)) = result {
                let error = std::mem::replace(e, invalid("consumed child refusal"));
                self.restore_parents();
                return Err(error);
            }
        }
        // Validate the entire batch before any cover insertion. Completed
        // child guards stay held beside every parent through this transaction.
        for i in 0..self.results.len() {
            let evaluation = self.results[i].as_ref().unwrap().as_ref().unwrap().clone();
            self.classify(&evaluation)?;
        }
        let expanded = self.parents.len();
        let old_recorded = self.ledger.recorded();
        let old_closed = self.closed.len();
        let old_seq = self.seq;
        let old_receipt = self.receipt;
        for i in 0..self.results.len() {
            let evaluation = self.results[i].take().unwrap()?;
            if let Err(error) = self.eval_finish(evaluation) {
                self.ledger.rollback_to(old_recorded);
                self.closed.truncate(old_closed);
                // Metadata-only undo is already priced by control_lease.
                // BinaryHeap::retain reuses its actual allocation.
                self.frontier.retain(|entry| entry.seq < old_seq);
                self.seq = old_seq;
                self.receipt.bound_leaves = old_receipt.bound_leaves;
                self.receipt.empty_leaves = old_receipt.empty_leaves;
                self.receipt.evaluated_domains = old_receipt.evaluated_domains;
                return Err(error);
            }
            if let Some(observer) = after_leaf.as_mut() {
                observer(self, i);
            }
        }
        self.receipt.expansions = add(self.receipt.expansions, expanded)?;
        self.parents.clear();
        self.children.clear();
        self.results.clear();
        Ok(None)
    }
    fn finish(self, reason: PreparedTreeOpenReason) -> Result<PreparedTreeOutcome> {
        self.checkpoint(None)?;
        if !self.root_pending && self.frontier.is_empty() && self.parents.is_empty() {
            let cover = self
                .ledger
                .prove(self.root.classes(), Some(self.deadline))
                .map_err(cover_error)?;
            Ok(PreparedTreeOutcome::Covered(PreparedTreeCertificate {
                state: self,
                cover,
            }))
        } else {
            Ok(PreparedTreeOutcome::Open(PreparedTreeOpen {
                state: self,
                reason,
            }))
        }
    }
}
/// Actual MarginRowBab typed entry. A source/ledger refusal cannot turn a
/// region into a closed leaf or discard its parent from the frontier.
pub(super) fn run(
    open: PreparedRootOpen,
    options: PreparedTreeOptions,
    requested: Option<Instant>,
) -> Result<PreparedTreeOutcome> {
    let mut state = PreparedTreeState::new(open, options, requested)?;
    match state.root_entry() {
        Ok(()) => {}
        Err(error) if recoverable(&error) => {
            return state.finish(PreparedTreeOpenReason::ResourceRefusal)
        }
        Err(error) => return Err(error),
    }
    while !state.frontier.is_empty() {
        match state.schedule_batch() {
            Ok(Some(reason)) => return state.finish(reason),
            Ok(None) => {}
            Err(e) if recoverable(&e) => {
                state.restore_parents();
                return state.finish(PreparedTreeOpenReason::ResourceRefusal);
            }
            Err(e) => return Err(e),
        }
    }
    state.finish(PreparedTreeOpenReason::ExpansionLimit)
}
// Debug exposes the actual existing claims separately. These snapshots do
// not authorize a source, certify current validity or measure physical heap.
impl fmt::Debug for PreparedTreeCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTreeCertificate")
            .field("target", &self.target())
            .field("classes", &self.classes())
            .field("receipt", &self.receipt())
            .field("cover_reserved_bytes", &self.cover.resident_bytes())
            .field(
                "cover_ledger_reserved_bytes",
                &self.state.ledger.resident_bytes(),
            )
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for PreparedTreeOpen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTreeOpen")
            .field("target", &self.target())
            .field("classes", &self.classes())
            .field("reason", &self.reason())
            .field("open_domains", &self.open_domains())
            .field("receipt", &self.receipt())
            .field(
                "cover_ledger_reserved_bytes",
                &self.state.ledger.resident_bytes(),
            )
            .finish_non_exhaustive()
    }
}
impl PreparedTreeCertificate {
    pub fn target(&self) -> usize {
        self.state.root.target()
    }
    pub fn classes(&self) -> &[usize] {
        self.state.root.classes()
    }
    pub(in crate::margin_row) fn source(
        &self,
    ) -> &crate::margin_row::support_program::PreparedRootSource {
        self.state.root.adapter().source()
    }

    pub fn receipt(&self) -> PreparedTreeReceipt {
        self.state.receipt
    }
    pub fn deadline(&self) -> Instant {
        self.state.deadline.min(self.cover.deadline())
    }
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.state.checkpoint(Some(
            requested.map_or(self.deadline(), |d| d.min(self.deadline())),
        ))?;
        self.cover.checkpoint(requested).map_err(cover_error)
    }
    pub(in crate::margin_row) fn root(&self) -> &PreparedDomain {
        &self.state.root
    }
    pub(crate) fn cover(&self, requested: Option<Instant>) -> Result<&TreeCover> {
        self.checkpoint(requested)?;
        self.cover.cover(requested).map_err(cover_error)
    }
    pub(in crate::margin_row) fn leaves(
        &self,
    ) -> impl Iterator<Item = (&PreparedDomainEvaluation, &PreparedDomainPath)> {
        self.state
            .closed
            .iter()
            .map(|leaf| (&leaf.evaluation, &leaf.path))
    }
    #[expect(dead_code, reason = "prepared collector integration")]
    pub(crate) fn ledger(
        &self,
        requested: Option<Instant>,
    ) -> Result<&crate::certify::CoverLedger> {
        self.checkpoint(requested)?;
        self.state.ledger.ledger(requested).map_err(cover_error)
    }
    pub(crate) fn budget(&self) -> &WorkBudget {
        self.state.root.budget()
    }
}
impl PreparedTreeOpen {
    /// Borrow the exact original session held by this resumable search. The
    /// caller must checkpoint the tree before using it for source binding.
    pub fn root_open(&self) -> &PreparedRootOpen {
        self.state.root.root_open()
    }
    pub fn target(&self) -> usize {
        self.state.root.target()
    }
    pub fn classes(&self) -> &[usize] {
        self.state.root.classes()
    }
    pub fn deadline(&self) -> Instant {
        self.state.deadline
    }
    #[expect(dead_code, reason = "prepared collector integration")]
    pub(in crate::margin_row) fn source(
        &self,
    ) -> &crate::margin_row::support_program::PreparedRootSource {
        self.state.root.adapter().source()
    }
    pub fn restricted(mut self, requested: Option<Instant>) -> Result<Self> {
        self.state.deadline = requested.map_or(self.state.deadline, |d| d.min(self.state.deadline));
        self.state.checkpoint(None)?;
        Ok(self)
    }

    pub fn receipt(&self) -> PreparedTreeReceipt {
        self.state.receipt
    }
    pub fn reason(&self) -> PreparedTreeOpenReason {
        self.reason
    }
    pub fn open_domains(&self) -> usize {
        self.state.frontier.len() + self.state.parents.len() + usize::from(self.state.root_pending)
    }
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.state.checkpoint(requested)
    }
    /// Resume consumes the same frontier/cache/claims and expansion limit.
    /// It grants no new allowance and cannot renew a completed producer clock.
    pub fn resume(mut self, requested: Option<Instant>) -> Result<PreparedTreeOutcome> {
        self.state.deadline = requested.map_or(self.state.deadline, |d| d.min(self.state.deadline));
        self.state.checkpoint(None)?;
        if self.state.root_pending {
            match self.state.root_entry() {
                Ok(()) => {}
                Err(error) if recoverable(&error) => {
                    return self.state.finish(PreparedTreeOpenReason::ResourceRefusal)
                }
                Err(error) => return Err(error),
            }
        }
        while !self.state.frontier.is_empty() {
            match self.state.schedule_batch() {
                Ok(Some(reason)) => return self.state.finish(reason),
                Ok(None) => {}
                Err(e) if recoverable(&e) => {
                    self.state.restore_parents();
                    return self.state.finish(PreparedTreeOpenReason::ResourceRefusal);
                }
                Err(e) => return Err(e),
            }
        }
        self.state.finish(PreparedTreeOpenReason::ExpansionLimit)
    }
}
#[cfg(test)]
#[path = "prepared_tree_tests.rs"]
mod tests;
