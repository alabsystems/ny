// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Request-owned storage for a future finite SoA collector.
//!
//! All slots and materialized relations share one live-payload account. Copies
//! retain their source's charge, reconcile Vec capacities before touching them,
//! and poll the original absolute deadline in bounded runs. These wrappers do
//! not enable the finite traversal or certify a supplied seed/bound map.

use super::{BatchSlot, CrownDispatchPlan};
use crate::amortise::{ChunkedPoll, ReductionPoll, POLL_CHUNK_UNITS};
use crate::bounds::patches::PatchesMaterializationDeadline;
use crate::certify::work::WorkLease;
use crate::network::crown_memory::cpu_crown_dense_budget_bytes;
use crate::LinearBounds;
use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SITE: &str = "bounded SoA storage";
const ENTRY: &str = "bounded SoA storage entry";
const AFTER_RESERVE: &str = "bounded SoA storage after reserve";
const FILL: &str = "bounded SoA storage fill";
const COPY: &str = "bounded SoA storage copy";
const PUBLISH: &str = "bounded SoA storage publication";
const SLOT_COMMIT: &str = "bounded SoA storage complete slot";
const SEED_COMMIT: &str = "bounded SoA storage complete seed";
const RELATION_COMMIT: &str = "bounded SoA storage complete relation";
const PENDING_COMMIT: &str = "bounded SoA storage complete pending";

fn invalid(detail: &str) -> NyError {
    NyError::InvalidSpec(format!("{SITE}: {detail}"))
}

fn narrower(authority: Option<Instant>, requested: Option<Instant>) -> Option<Instant> {
    match (authority, requested) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

struct Poll {
    shared: PatchesMaterializationDeadline,
    query: Option<Arc<bounded_query::QueryAuthority>>,
    query_since_check: usize,
    #[cfg(test)]
    forced_stage: Option<&'static str>,
}

impl Poll {
    fn new(deadline: Option<Instant>) -> Self {
        Self {
            shared: PatchesMaterializationDeadline::new(deadline),
            query: None,
            query_since_check: 0,
            #[cfg(test)]
            forced_stage: None,
        }
    }

    fn for_owner(owner: &Arc<SoaStorageBudget>, requested: Option<Instant>) -> Self {
        let mut poll = Self::new(narrower(owner.deadline, requested));
        poll.query = owner.query.clone();
        poll
    }

    /// Admit the complete bounded loop before touching any value. This is
    /// separate from after-work deadline accounting; it never debits late.
    fn admit_elements(&self, count: usize, units: usize, _stage: &'static str) -> Result<()> {
        if let Some(query) = &self.query {
            let work = count
                .checked_mul(units)
                .ok_or_else(|| invalid("query primitive work overflow"))?;
            query.work(work)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self, stage: &'static str) -> Result<()> {
        if let Some(query) = &self.query {
            query.checkpoint()?;
            self.query_since_check = 0;
        }
        #[cfg(test)]
        if self.forced_stage == Some(stage) {
            return Err(NyError::DeadlineExceeded(format!(
                "{SITE}: forced test expiry at {stage}"
            )));
        }
        self.shared.checkpoint(stage)
    }

    fn work(&mut self, units: usize, stage: &'static str) -> Result<()> {
        #[cfg(test)]
        if self.forced_stage == Some(stage) {
            return self.checkpoint(stage);
        }
        self.shared.work(units, stage)?;
        if let Some(query) = &self.query {
            self.query_since_check = self.query_since_check.saturating_add(units);
            if self.query_since_check >= POLL_CHUNK_UNITS {
                query.checkpoint()?;
                self.query_since_check = 0;
            }
        }
        Ok(())
    }
}

/// One shared account for declared retained payload plus every owned Vec capacity.
/// External graph/maps/seed state kept across calls belongs in `retained_bytes`.
/// This is not a census of allocator metadata or unobservable ndarray slack;
/// the existing process-envelope policy still limits the total dense allowance.
pub(super) struct SoaStorageBudget {
    retained_bytes: usize,
    ceiling_bytes: usize,
    reserved_bytes: AtomicUsize,
    deadline: Option<Instant>,
    query: Option<Arc<bounded_query::QueryAuthority>>,
}

impl SoaStorageBudget {
    pub(super) fn new(retained_bytes: usize, deadline: Option<Instant>) -> Result<Arc<Self>> {
        let mut poll = Poll::new(deadline);
        poll.checkpoint(ENTRY)?;
        let ceiling_bytes = cpu_crown_dense_budget_bytes();
        if retained_bytes > ceiling_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: retained_bytes,
                budget_bytes: ceiling_bytes,
                site: SITE,
            });
        }
        let result = Arc::new(Self {
            retained_bytes,
            ceiling_bytes,
            reserved_bytes: AtomicUsize::new(0),
            deadline,
            query: None,
        });
        poll.checkpoint(PUBLISH)?;
        Ok(result)
    }

    fn memory_error(&self, required_bytes: usize) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: self.ceiling_bytes.min(cpu_crown_dense_budget_bytes()),
            site: SITE,
        }
    }

    fn charge(self: &Arc<Self>, bytes: usize, poll: &mut Poll) -> Result<Charge> {
        poll.admit_elements(1, 64, ENTRY)?;
        self.add(bytes, poll)?;
        let mut charge = Charge {
            owner: Arc::clone(self),
            bytes,
            query_lease: None,
        };
        if let Some(query) = &self.query {
            charge.query_lease = Some(query.bytes(bytes)?);
        }
        Ok(charge)
    }

    fn add(&self, bytes: usize, poll: &mut Poll) -> Result<()> {
        let mut old = self.reserved_bytes.load(Ordering::Acquire);
        loop {
            // Also bound contention: a failed CAS cannot spin past authority.
            poll.checkpoint(ENTRY)?;
            let next = old
                .checked_add(bytes)
                .ok_or_else(|| self.memory_error(usize::MAX))?;
            let required = self
                .retained_bytes
                .checked_add(next)
                .ok_or_else(|| self.memory_error(usize::MAX))?;
            if required > self.ceiling_bytes.min(cpu_crown_dense_budget_bytes()) {
                return Err(self.memory_error(required));
            }
            match self.reserved_bytes.compare_exchange_weak(
                old,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => old = observed,
            }
        }
    }

    fn check_live(&self, poll: &mut Poll) -> Result<()> {
        poll.checkpoint(PUBLISH)?;
        let required = self
            .retained_bytes
            .checked_add(self.reserved_bytes.load(Ordering::Acquire))
            .ok_or_else(|| self.memory_error(usize::MAX))?;
        if required > self.ceiling_bytes.min(cpu_crown_dense_budget_bytes()) {
            return Err(self.memory_error(required));
        }
        Ok(())
    }
}

struct Charge {
    owner: Arc<SoaStorageBudget>,
    bytes: usize,
    query_lease: Option<WorkLease>,
}

impl Charge {
    fn reconcile<T>(&mut self, requested: usize, capacity: usize, poll: &mut Poll) -> Result<()> {
        let extra = capacity
            .saturating_sub(requested)
            .checked_mul(size_of::<T>())
            .ok_or_else(|| self.owner.memory_error(usize::MAX))?;
        let total = self
            .bytes
            .checked_add(extra)
            .ok_or_else(|| self.owner.memory_error(usize::MAX))?;
        self.reconcile_query(extra)?;
        self.owner.add(extra, poll)?;
        self.bytes = total;
        Ok(())
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        let old = self
            .owner
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(old >= self.bytes);
    }
}

fn reserve<T>(length: usize, charge: &mut Charge, poll: &mut Poll) -> Result<Vec<T>> {
    poll.checkpoint(ENTRY)?;
    let mut out = Vec::new();
    out.try_reserve_exact(length).map_err(|_| {
        charge.owner.memory_error(
            charge
                .owner
                .retained_bytes
                .saturating_add(charge.owner.reserved_bytes.load(Ordering::Acquire)),
        )
    })?;
    charge.reconcile::<T>(length, out.capacity(), poll)?;
    poll.checkpoint(AFTER_RESERVE)?;
    Ok(out)
}

// Metadata slots are larger than one scalar. Price their complete element
// width so a fill never treats 4096 whole slot headers as 4096 float touches.
fn element_units<T>() -> usize {
    size_of::<T>().div_ceil(size_of::<f32>()).max(1)
}

fn chunk_elements<T>() -> usize {
    (POLL_CHUNK_UNITS / element_units::<T>()).max(1)
}

fn filled<T: Copy>(
    length: usize,
    value: T,
    charge: &mut Charge,
    poll: &mut Poll,
) -> Result<Vec<T>> {
    let mut out = reserve(length, charge, poll)?;
    poll.admit_elements(length, element_units::<T>(), FILL)?;
    while out.len() < length {
        poll.checkpoint(ENTRY)?;
        let count = (length - out.len()).min(chunk_elements::<T>());
        out.resize(out.len() + count, value);
        poll.work(count * element_units::<T>(), FILL)?;
    }
    poll.checkpoint(PUBLISH)?;
    Ok(out)
}

fn copied(source: &[f32], charge: &mut Charge, poll: &mut Poll) -> Result<Vec<f32>> {
    let mut out = reserve(source.len(), charge, poll)?;
    poll.admit_elements(source.len(), 1, COPY)?;
    crate::amortise::extend_polled(&mut out, source, |units| poll.work(units, COPY))?;
    poll.checkpoint(PUBLISH)?;
    Ok(out)
}

fn copy_into(
    destination: &mut [f32],
    contiguous: Option<&[f32]>,
    source: impl Iterator<Item = f32>,
    poll: &mut Poll,
) -> Result<()> {
    poll.checkpoint(ENTRY)?;
    poll.admit_elements(destination.len().saturating_add(1), 1, COPY)?;
    if let Some(source) = contiguous {
        if source.len() != destination.len() {
            return Err(invalid("seed slice disagrees with its shape"));
        }
        for (out, source) in destination
            .chunks_mut(POLL_CHUNK_UNITS)
            .zip(source.chunks(POLL_CHUNK_UNITS))
        {
            out.copy_from_slice(source);
            poll.work(out.len(), COPY)?;
        }
        return poll.checkpoint(PUBLISH);
    }
    let mut source = source;
    let mut amortised = ChunkedPoll::new(|units| poll.work(units, COPY));
    for value in destination {
        *value = source
            .next()
            .ok_or_else(|| invalid("seed payload is shorter than its shape"))?;
        amortised.tick()?;
    }
    amortised.flush()?;
    if source.next().is_some() {
        return Err(invalid("seed payload is longer than its shape"));
    }
    poll.checkpoint(PUBLISH)
}

fn slot_layout(domains: usize, rows: usize, cols: usize) -> Result<(usize, usize, usize)> {
    if domains == 0 || rows == 0 || cols == 0 {
        return Err(invalid("slot dimensions must be nonzero"));
    }
    let overflow = || invalid("slot shape or byte count overflows usize");
    let coefficients = domains
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(cols))
        .ok_or_else(overflow)?;
    let biases = domains.checked_mul(rows).ok_or_else(overflow)?;
    let payload = coefficients
        .checked_mul(4)
        .and_then(|n| biases.checked_mul(2).and_then(|b| n.checked_add(b)))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| domains.checked_mul(3).and_then(|m| n.checked_add(m)))
        .ok_or_else(overflow)?;
    Ok((coefficients, biases, payload))
}

/// A slot and its live payload lease always travel together.
/// Returning `&BatchSlot` permits kernel reads without releasing the lease.
pub(super) struct BoundedBatchSlot {
    storage: BatchSlot,
    charge: Charge,
    deadline: Option<Instant>,
}

impl BoundedBatchSlot {
    fn zeroed_with_poll(
        owner: &Arc<SoaStorageBudget>,
        domains: usize,
        rows: usize,
        cols: usize,
        deadline: Option<Instant>,
        poll: &mut Poll,
    ) -> Result<Self> {
        poll.checkpoint(ENTRY)?;
        let (coefficients, biases, bytes) = slot_layout(domains, rows, cols)?;
        let mut charge = owner.charge(bytes, poll)?;
        let storage = BatchSlot {
            rows,
            cols,
            active: filled(domains, false, &mut charge, poll)?,
            lower_a: filled(coefficients, 0.0, &mut charge, poll)?,
            upper_a: filled(coefficients, 0.0, &mut charge, poll)?,
            lower_b: filled(biases, 0.0, &mut charge, poll)?,
            upper_b: filled(biases, 0.0, &mut charge, poll)?,
            lower_a_err: filled(coefficients, 0.0, &mut charge, poll)?,
            upper_a_err: filled(coefficients, 0.0, &mut charge, poll)?,
            lower_err_present: filled(domains, false, &mut charge, poll)?,
            upper_err_present: filled(domains, false, &mut charge, poll)?,
        };
        owner.check_live(poll)?;
        charge.materialize()?;
        poll.checkpoint(SLOT_COMMIT)?;
        Ok(Self {
            storage,
            charge,
            deadline,
        })
    }

    pub(super) fn try_zeroed(
        owner: &Arc<SoaStorageBudget>,
        domains: usize,
        rows: usize,
        cols: usize,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let deadline = narrower(owner.deadline, requested);
        Self::zeroed_with_poll(
            owner,
            domains,
            rows,
            cols,
            deadline,
            &mut Poll::for_owner(owner, deadline),
        )
    }

    fn seeded_with_poll(
        owner: &Arc<SoaStorageBudget>,
        domains: usize,
        seed: &LinearBounds,
        deadline: Option<Instant>,
        poll: &mut Poll,
    ) -> Result<Self> {
        poll.checkpoint(ENTRY)?;
        let rows = seed.num_outputs();
        let cols = seed.num_inputs();
        if seed.upper_a().dim() != seed.lower_a().dim()
            || seed.lower_b().len() != rows
            || seed.upper_b().len() != rows
            || seed.lower_a_err().is_some()
            || seed.upper_a_err().is_some()
        {
            return Err(invalid("seed must retain the original plain-spec contract"));
        }
        // In addition to externally retained state, price the borrowed seed
        // while staging. No unaccounted temporary per-plane Vec is created.
        let _source = owner.charge(seed.memory_bytes(), poll)?;
        let mut out = Self::zeroed_with_poll(owner, domains, rows, cols, deadline, poll)?;
        let rc = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("seed shape overflow"))?;
        poll.admit_elements(domains, 1, COPY)?;
        for domain in 0..domains {
            copy_into(
                &mut out.storage.lower_a[domain * rc..(domain + 1) * rc],
                seed.lower_a().as_slice(),
                seed.lower_a().iter().copied(),
                poll,
            )?;
            copy_into(
                &mut out.storage.upper_a[domain * rc..(domain + 1) * rc],
                seed.upper_a().as_slice(),
                seed.upper_a().iter().copied(),
                poll,
            )?;
            copy_into(
                &mut out.storage.lower_b[domain * rows..(domain + 1) * rows],
                seed.lower_b().as_slice(),
                seed.lower_b().iter().copied(),
                poll,
            )?;
            copy_into(
                &mut out.storage.upper_b[domain * rows..(domain + 1) * rows],
                seed.upper_b().as_slice(),
                seed.upper_b().iter().copied(),
                poll,
            )?;
            out.storage.active[domain] = true;
            poll.work(1, COPY)?;
        }
        owner.check_live(poll)?;
        poll.checkpoint(SEED_COMMIT)?;
        Ok(out)
    }

    pub(super) fn try_seeded(
        owner: &Arc<SoaStorageBudget>,
        domains: usize,
        seed: &LinearBounds,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let deadline = narrower(owner.deadline, requested);
        Self::seeded_with_poll(
            owner,
            domains,
            seed,
            deadline,
            &mut Poll::for_owner(owner, deadline),
        )
    }

    pub(super) fn slot(&self) -> &BatchSlot {
        &self.storage
    }

    fn materialize_with_poll(
        &self,
        domain: usize,
        deadline: Option<Instant>,
        poll: &mut Poll,
    ) -> Result<Option<BoundedLinearBounds>> {
        poll.checkpoint(ENTRY)?;
        let storage = &self.storage;
        let active = *storage
            .active
            .get(domain)
            .ok_or_else(|| invalid("domain index exceeds slot"))?;
        if !active {
            return Ok(None);
        }
        let (coefficients, biases, _) =
            slot_layout(storage.active.len(), storage.rows, storage.cols)?;
        if [
            &storage.lower_a,
            &storage.upper_a,
            &storage.lower_a_err,
            &storage.upper_a_err,
        ]
        .iter()
        .any(|p| p.len() != coefficients)
            || [&storage.lower_b, &storage.upper_b]
                .iter()
                .any(|p| p.len() != biases)
            || storage.lower_err_present.len() != storage.active.len()
            || storage.upper_err_present.len() != storage.active.len()
        {
            return Err(invalid("slot plane or marker layout mismatch"));
        }
        let rc = storage.rows * storage.cols;
        let lo = domain * rc;
        let hi = lo + rc;
        let blo = domain * storage.rows;
        let bhi = blo + storage.rows;
        let planes = 2
            + usize::from(storage.lower_err_present[domain])
            + usize::from(storage.upper_err_present[domain]);
        let bytes = rc
            .checked_mul(planes)
            .and_then(|n| storage.rows.checked_mul(2).and_then(|b| n.checked_add(b)))
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| invalid("materialized byte count overflow"))?;
        let mut charge = self.charge.owner.charge(bytes, poll)?;
        let matrix =
            |values: &[f32], charge: &mut Charge, poll: &mut Poll| -> Result<Array2<f32>> {
                Array2::from_shape_vec((storage.rows, storage.cols), copied(values, charge, poll)?)
                    .map_err(|_| invalid("copied matrix shape mismatch"))
            };
        let bounds = LinearBounds {
            lower_a: matrix(&storage.lower_a[lo..hi], &mut charge, poll)?,
            upper_a: matrix(&storage.upper_a[lo..hi], &mut charge, poll)?,
            lower_b: Array1::from_vec(copied(&storage.lower_b[blo..bhi], &mut charge, poll)?),
            upper_b: Array1::from_vec(copied(&storage.upper_b[blo..bhi], &mut charge, poll)?),
            lower_a_err: if storage.lower_err_present[domain] {
                Some(matrix(&storage.lower_a_err[lo..hi], &mut charge, poll)?)
            } else {
                None
            },
            upper_a_err: if storage.upper_err_present[domain] {
                Some(matrix(&storage.upper_a_err[lo..hi], &mut charge, poll)?)
            } else {
                None
            },
        };
        self.charge.owner.check_live(poll)?;
        charge.materialize()?;
        poll.checkpoint(RELATION_COMMIT)?;
        Ok(Some(BoundedLinearBounds {
            bounds,
            _charge: charge,
            deadline,
        }))
    }

    pub(super) fn try_materialize(
        &self,
        domain: usize,
        requested: Option<Instant>,
    ) -> Result<Option<BoundedLinearBounds>> {
        let deadline = narrower(self.deadline, requested);
        self.materialize_with_poll(
            domain,
            deadline,
            &mut Poll::for_owner(&self.charge.owner, deadline),
        )
    }
}

/// Keeps the materialized relation charged while a borrowed shared kernel reads it.
pub(super) struct BoundedLinearBounds {
    bounds: LinearBounds,
    _charge: Charge,
    deadline: Option<Instant>,
}

impl BoundedLinearBounds {
    pub(super) fn bounds(&self) -> &LinearBounds {
        &self.bounds
    }
    pub(super) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

mod relation_sealed {
    pub trait Sealed {}
}

/// Borrow a relation while its concrete owner retains every payload claim.
/// Only the original materialization and consuming kernel-output wrapper can
/// implement this interface; it creates no allocation or source provenance.
pub(super) trait BoundedRelation: relation_sealed::Sealed {
    fn bounds(&self) -> &LinearBounds;
    fn deadline(&self) -> Option<Instant>;
    fn owner(&self) -> &Arc<SoaStorageBudget>;
}

impl relation_sealed::Sealed for BoundedLinearBounds {}
impl BoundedRelation for BoundedLinearBounds {
    fn bounds(&self) -> &LinearBounds {
        self.bounds()
    }
    fn deadline(&self) -> Option<Instant> {
        self.deadline()
    }
    fn owner(&self) -> &Arc<SoaStorageBudget> {
        &self._charge.owner
    }
}

/// A private pending transaction. Fallible seed publication consumes the owner;
/// an error discards every partial slot and releases its leases before fallback.
pub(super) struct BoundedBatchPending {
    slots: Vec<Option<BoundedBatchSlot>>,
    input_accumulated: Vec<bool>,
    network_input_idx: usize,
    domains: usize,
    charge: Charge,
    deadline: Option<Instant>,
}

impl BoundedBatchPending {
    pub(super) fn try_new(
        plan: &CrownDispatchPlan,
        domains: usize,
        owner: &Arc<SoaStorageBudget>,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let deadline = narrower(owner.deadline, requested);
        let mut poll = Poll::for_owner(owner, deadline);
        poll.checkpoint(ENTRY)?;
        if domains == 0 {
            return Err(invalid("pending batch must have domains"));
        }
        let count = plan
            .node_count()
            .checked_add(1)
            .ok_or_else(|| invalid("pending slot count overflow"))?;
        let bytes = count
            .checked_mul(size_of::<Option<BoundedBatchSlot>>())
            .and_then(|n| n.checked_add(domains))
            .ok_or_else(|| invalid("pending metadata bytes overflow"))?;
        let mut charge = owner.charge(bytes, &mut poll)?;
        let mut slots = reserve(count, &mut charge, &mut poll)?;
        poll.admit_elements(count, element_units::<Option<BoundedBatchSlot>>(), FILL)?;
        while slots.len() < count {
            poll.checkpoint(ENTRY)?;
            let chunk_len = (count - slots.len()).min(chunk_elements::<Option<BoundedBatchSlot>>());
            let end = slots.len() + chunk_len;
            while slots.len() < end {
                slots.push(None);
            }
            poll.work(
                chunk_len * element_units::<Option<BoundedBatchSlot>>(),
                FILL,
            )?;
        }
        let input_accumulated = filled(domains, false, &mut charge, &mut poll)?;
        owner.check_live(&mut poll)?;
        charge.materialize()?;
        Ok(Self {
            slots,
            input_accumulated,
            network_input_idx: plan.network_input_idx,
            domains,
            charge,
            deadline,
        })
    }

    fn seed_with_poll(mut self, idx: usize, seed: &LinearBounds, poll: &mut Poll) -> Result<Self> {
        poll.checkpoint(ENTRY)?;
        if idx >= self.slots.len() || self.slots[idx].is_some() {
            return Err(invalid("seed slot is unknown or already populated"));
        }
        let slot = BoundedBatchSlot::seeded_with_poll(
            &self.charge.owner,
            self.domains,
            seed,
            self.deadline,
            poll,
        )?;
        if idx == self.network_input_idx {
            poll.admit_elements(self.input_accumulated.len(), 1, FILL)?;
            for chunk in self.input_accumulated.chunks_mut(POLL_CHUNK_UNITS) {
                chunk.fill(true);
                poll.work(chunk.len(), FILL)?;
            }
        }
        self.slots[idx] = Some(slot);
        self.charge.owner.check_live(poll)?;
        poll.checkpoint(PENDING_COMMIT)?;
        Ok(self)
    }

    pub(super) fn seed_broadcast(self, idx: usize, seed: &LinearBounds) -> Result<Self> {
        let deadline = self.deadline;
        let mut poll = Poll::for_owner(&self.charge.owner, deadline);
        self.seed_with_poll(idx, seed, &mut poll)
    }

    pub(super) fn take(mut self, idx: usize) -> Result<(Self, Option<BoundedBatchSlot>)> {
        let mut poll = Poll::for_owner(&self.charge.owner, self.deadline);
        poll.checkpoint(ENTRY)?;
        poll.admit_elements(1, 8, ENTRY)?;
        let slot = self
            .slots
            .get_mut(idx)
            .ok_or_else(|| invalid("take slot is unknown"))?;
        let mut taken = slot.take();
        if let Some(slot) = &mut taken {
            slot.deadline = narrower(slot.deadline, self.deadline);
        }
        self.charge.owner.check_live(&mut poll)?;
        Ok((self, taken))
    }

    pub(super) fn input_accumulated(&self) -> &[bool] {
        &self.input_accumulated
    }
}

#[cfg(test)]
#[path = "bounded_storage_tests.rs"]
mod tests;

// Bounded merge preparation. Removing this block recovers the complete
// previously prepared storage module byte-for-byte.
#[path = "bounded_merge.rs"]
mod bounded_merge;

impl BoundedBatchPending {
    /// Consume the private transaction so a partial merge can never escape on
    /// a deadline, shape or live-budget refusal. Incoming storage stays leased.
    pub(super) fn accumulate_domain(
        mut self,
        idx: usize,
        domain: usize,
        contribution: &impl BoundedRelation,
    ) -> Result<Self> {
        let deadline = narrower(self.deadline, contribution.deadline());
        self.deadline = deadline;
        let mut poll = Poll::for_owner(&self.charge.owner, deadline);
        bounded_merge::accumulate(self, idx, domain, contribution, &mut poll)
    }
}

// Kernel storage preparation; shared numerical kernels are not wired to it yet.
#[path = "bounded_matrix.rs"]
mod bounded_matrix;

#[path = "bounded_query.rs"]
mod bounded_query;
