// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Irreversible early funding for ONE existing full/partial component row.
//! This is an allocation policy, not another proof kernel or renewed clock.
use super::*;
use std::sync::Mutex;

pub(super) const WORK: usize = 4_000_000;
const PREFLIGHT: usize = 4096;
const TICKET_BYTES: usize = 4096;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Policy {
    Legacy,
    OneJoint,
    RootCoupling,
    RootPortfolio,
    RootPortfolioAllPhases,
    RootPortfolioPhaseCube,
}
#[derive(Debug, Default)]
struct Use {
    end: Option<Instant>,
    checker_claimed: bool,
    work: usize,
    bytes: usize,
}
#[derive(Debug)]
pub(super) struct Ticket {
    source: usize,
    dimensions: (usize, usize, usize),
    components: usize,
    ledger: Arc<Mutex<Counts>>,
    clocks: (Instant, Instant),
    base_bytes: usize,
    payload: usize,
    used: Mutex<Use>,
}
/// O(1) metadata only. Every final valid plan has the same negative partition,
/// <=2P positive appearances, <=16 positives/full pair and <=32/partial pair.
pub(super) fn payload(d: usize, h: usize, positive: usize, components: usize) -> Option<usize> {
    if d == 0
        || d > 128
        || h == 0
        || h > 512
        || !(2..=64).contains(&components)
        || components > h
        || positive == 0
        || positive > h
    {
        return None;
    }
    let width = d.checked_add(1)?;
    let setup = components
        .checked_mul(width)?
        .checked_add(h)?
        .checked_add(1)?
        .checked_mul(32)?
        .checked_add(components.checked_mul(components)?.checked_mul(96)?)?
        .checked_add(4096)?;
    let full_positive = positive.checked_mul(2)?.min(16);
    let partial_positive = positive.checked_mul(2)?.min(32);
    let full = 256usize
        .checked_mul(full_positive)?
        .checked_mul(32)?
        .checked_add(
            h.checked_add(d)?
                .checked_add(full_positive)?
                .checked_add(24)?
                .checked_mul(256)?,
        )?
        .checked_add(8192)?;
    let partial = 8usize
        .checked_mul(partial_positive)?
        .checked_mul(32)?
        .checked_add(
            h.checked_add(d)?
                .checked_add(partial_positive)?
                .checked_add(32)?
                .checked_mul(256)?,
        )?
        .checked_add(width.checked_mul(1024)?)?
        .checked_add(8192)?;
    setup
        .checked_add(full.max(partial))?
        .checked_add(TICKET_BYTES)
}
/// Called after the checked baseline, two publication holds and component/phase
/// admission, BEFORE any row clones/refinement. A refused quote keeps preflight
/// debt; admitted credit is never returned, even if publication cannot happen.
pub(super) fn admit(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &mut Budget) -> bool {
    if b.prepaid.is_some() {
        return false;
    }
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.joint.prepaid_requested = true;
    }
    if b.charge(PREFLIGHT).is_none() {
        return false;
    }
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.joint.prepaid_work_units = PREFLIGHT;
        c.component_rows.joint.prepaid_used_work_units = PREFLIGHT;
    }
    let Some(bytes) = payload(
        b.dimensions.0,
        b.dimensions.1,
        b.dimensions.2,
        plan.components.len(),
    ) else {
        return false;
    };
    if b.source != std::ptr::from_ref(cuts.problem) as usize || !live(b.search_end) {
        return false;
    }
    let Ok(mut c) = b.state.lock() else {
        return false;
    };
    if !c.component_rows.admitted || c.replay_attempts != 0 || c.checks >= MAX_CHECKS {
        return false;
    }
    let Some(work) = c.work.checked_add(WORK - PREFLIGHT) else {
        return false;
    };
    let Some(admitted) = b.admitted_bytes.checked_add(bytes) else {
        return false;
    };
    let total_work = work
        .checked_add(c.reserved)
        .and_then(|v| v.checked_add(c.component_reserved));
    let total_bytes = admitted
        .checked_add(c.shared_bytes)
        .and_then(|v| v.checked_add(c.direction.cache_bytes))
        .and_then(|v| v.checked_add(c.child.cache_bytes))
        .and_then(|v| v.checked_add(c.component_rows.admitted_bytes));
    if total_work.is_none_or(|v| v > MAX_WORK) || total_bytes.is_none_or(|v| v > MAX_BYTES) {
        return false;
    }
    let ticket = Arc::new(Ticket {
        source: b.source,
        dimensions: b.dimensions,
        components: plan.components.len(),
        ledger: b.state.clone(),
        clocks: (b.authority, b.search_end),
        base_bytes: b.admitted_bytes,
        payload: bytes,
        used: Mutex::new(Use::default()),
    });
    c.work = work;
    c.checks += 1;
    let r = &mut c.component_rows.joint;
    r.prepaid_admitted = true;
    r.prepaid_work_units = WORK;
    r.prepaid_bytes = bytes;
    r.prepaid_used_work_units = PREFLIGHT;
    r.prepaid_checker_slots = 1;
    drop(c);
    b.admitted_bytes = admitted;
    b.prepaid = Some(ticket);
    true
}
impl Ticket {
    pub(super) fn owned(&self, row: &SignedMajorant) -> bool {
        self.source == row.source
            && self.source == row.budget.source
            && self.dimensions == row.budget.dimensions
            && self.components == row.plan.components.len()
            && self.clocks == (row.budget.authority, row.budget.search_end)
            && Arc::ptr_eq(&self.ledger, &row.budget.state)
            && self.base_bytes.checked_add(self.payload) == Some(row.budget.admitted_bytes)
    }
    /// The original first-publication window is latched once; callers cannot
    /// renew it by passing a later end. Failure after this point is final.
    pub(super) fn begin(&self, row: &SignedMajorant, end: Instant) -> bool {
        if !self.owned(row) {
            return false;
        }
        let Ok(mut use_) = self.used.lock() else {
            return false;
        };
        if use_.end.is_some() {
            return false;
        }
        use_.end = Some(end.min(self.clocks.0).min(self.clocks.1));
        live(use_.end.unwrap())
    }
    pub(super) fn end(&self, row: &SignedMajorant, end: Instant) -> Option<Instant> {
        if !self.owned(row) {
            return None;
        }
        Some(
            end.min(self.used.lock().ok()?.end?)
                .min(self.clocks.0)
                .min(self.clocks.1),
        )
    }
    /// One fresh producer/check capability, independent of the outer appender.
    /// A second private producer call refuses BEFORE repeating choice scans.
    pub(super) fn claim_checker(&self, row: &SignedMajorant, end: Instant) -> bool {
        if !self.owned(row) {
            return false;
        }
        let Ok(mut use_) = self.used.lock() else {
            return false;
        };
        let Some(window) = use_.end else {
            return false;
        };
        if use_.checker_claimed {
            return false;
        }
        use_.checker_claimed = true;
        live(end.min(window).min(self.clocks.0).min(self.clocks.1))
    }
    pub(super) fn remaining(&self, row: &SignedMajorant) -> Option<(usize, usize)> {
        if !self.owned(row) {
            return None;
        }
        let use_ = self.used.lock().ok()?;
        live(use_.end?).then_some((
            WORK.checked_sub(PREFLIGHT)?.checked_sub(use_.work)?,
            self.payload
                .checked_sub(TICKET_BYTES)?
                .checked_sub(use_.bytes)?,
        ))
    }
    /// This records consumption ONLY. Global work/payload were debited before
    /// refinement, so no global counter or component payload is added again.
    pub(super) fn pay(
        &self,
        row: &SignedMajorant,
        work: usize,
        bytes: usize,
        end: Instant,
    ) -> Option<()> {
        if !self.owned(row) {
            return None;
        }
        let mut use_ = self.used.lock().ok()?;
        if !live(end.min(use_.end?).min(self.clocks.0).min(self.clocks.1)) {
            return None;
        }
        let next_work = use_.work.checked_add(work)?;
        let next_bytes = use_.bytes.checked_add(bytes)?;
        if next_work > WORK.checked_sub(PREFLIGHT)?
            || next_bytes > self.payload.checked_sub(TICKET_BYTES)?
        {
            return None;
        }
        let mut c = self.ledger.lock().ok()?;
        let component_work = c.component_rows.work_units.checked_add(work)?;
        let r = &mut c.component_rows.joint;
        let used = r.work_units.checked_add(work)?;
        let payload = r.admitted_bytes.checked_add(bytes)?;
        r.work_units = used;
        r.admitted_bytes = payload;
        r.prepaid_used_work_units = PREFLIGHT.checked_add(next_work)?;
        c.component_rows.work_units = component_work;
        use_.work = next_work;
        use_.bytes = next_bytes;
        Some(())
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_prepaid_joint_default_tests.rs"]
mod default_tests;
