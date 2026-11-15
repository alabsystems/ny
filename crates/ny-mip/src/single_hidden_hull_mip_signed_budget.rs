// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! A separate, explicitly additive 64M engineering ledger; no renewed clock.
use super::*;
use std::sync::{Arc, Mutex};
pub(super) const MAX_WORK: usize = 64_000_000;
pub(super) const MAX_BYTES: usize = 8 << 20;
pub(super) const MAX_MASKS: usize = 4096;
pub(super) const MAX_POSITIVE: usize = 16;
pub(super) const MAX_CHECKS: usize = 8;
#[derive(Debug, Default)]
pub(super) struct Counts {
    pub direction: SignedDirectionReport,
    pub root_coupling: SignedRootCouplingReport,
    pub mass: SignedMassReport,
    pub portfolio: portfolio::State,
    pub phase_all: phase::AllState,
    pub phase_cube: phase::cube::State,
    // EXTERNAL DRAFT: no Admission constructor exists until additive funding.
    pub native_rows: native_publication::State,
    pub component_rows: SignedComponentRowsReport,
    pub component_reserved: usize,
    pub child: SignedChildReport,
    pub corner: corners::State,
    pub prepaid_collection_work: usize,
    pub prepaid_bind_work: usize,
    pub work: usize,
    pub reserved: usize,
    pub shared_bytes: usize,
    pub sharing_attempted: bool,
    pub checks: usize,
    pub replay_attempts: usize,
    pub source_replays: usize,
    pub completed_masks: usize,
    pub checked_masks: usize,
    pub coordinate_proposals: usize,
    pub refined_components: usize,
    pub reason: Option<&'static str>,
}
#[derive(Debug, Clone)]
pub(super) struct Budget {
    pub source: usize,
    pub dimensions: (usize, usize, usize),
    pub authority: Instant,
    pub search_end: Instant,
    pub admitted_bytes: usize,
    pub state: Arc<Mutex<Counts>>,
    pub portfolio_enabled: bool,
    pub portfolio_phase_enabled: bool,
    pub phase_all_enabled: bool,
    pub phase_cube_enabled: bool,
    pub prepaid: Option<Arc<prepaid::Ticket>>,
    pub root_coupling: Option<Arc<root_coupling::Ticket>>,
}
/// Only this module's constructor can mint the pre-escape stage capability.
/// Capture owns another ledger Arc; Arc strong counts cannot prove this stage.
pub(super) struct NativeConstruction {
    source: usize,
    ledger: *const Mutex<Counts>,
    clocks: (Instant, Instant),
}
impl NativeConstruction {
    pub(super) fn matches(&self, b: &Budget) -> bool {
        self.source == b.source
            && self.ledger == Arc::as_ptr(&b.state)
            && self.clocks == (b.authority, b.search_end)
    }
}
impl Budget {
    pub(super) fn new(cuts: &HullRootMipCuts<'_>, end: Instant, reserve: Duration) -> Option<Self> {
        Self::new_mode(cuts, end, reserve, NativePolicy::Disabled)
    }
    pub(super) fn new_with_native_rows(
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        reserve: Duration,
    ) -> Option<Self> {
        Self::new_mode(cuts, end, reserve, NativePolicy::Rows)
    }
    pub(super) fn new_with_portfolio_conditions(
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        reserve: Duration,
    ) -> Option<Self> {
        Self::new_mode(cuts, end, reserve, NativePolicy::Conditional)
    }
    fn new_mode(
        cuts: &HullRootMipCuts<'_>,
        end: Instant,
        reserve: Duration,
        native: NativePolicy,
    ) -> Option<Self> {
        shape(cuts)?;
        let p = cuts.problem;
        let end = end.min(cuts.deadline);
        let authority = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
        let search_end = authority
            .checked_sub(reserve.min(Duration::from_secs(2)))?
            .checked_sub(Duration::from_millis(75))?;
        if !live(search_end) {
            return None;
        }
        let cells = p.input_dim.checked_mul(p.hidden_dim)?;
        let positive = p.c.iter().filter(|c| **c > 0.).count();
        // Simultaneous point/witness/bit-seal copies and bounded scratch.
        // This is conservative nominal Vec payload, not allocator overhead/RSS.
        let capture = cuts.lp_primal_report();
        let bytes = cells
            .checked_mul(64)?
            .checked_add(positive.checked_mul(256)?.checked_mul(48)?)?
            .checked_add((p.hidden_dim + p.input_dim + 1).checked_mul(512)?)?
            .checked_add(search::scratch_bytes(p.input_dim)?)?
            .checked_add(capture.peak_bytes)?;
        // New fixed inline Counts field is not free on disabled/refused paths.
        // This is an explicit compatibility change to the baseline byte quote.
        let bytes = bytes
            .checked_add(size_of::<native_publication::State>())?
            .checked_add(align_of::<native_publication::State>())?
            .checked_add(phase::cube::cell::FIXED_METADATA_BYTES)?
            .checked_add(2 * size_of::<usize>())?
            .checked_add(corners::FIXED_BYTES)?;
        if bytes > MAX_BYTES {
            return None;
        }
        let mut budget = Self {
            source: std::ptr::from_ref(cuts.problem) as usize,
            dimensions: (p.input_dim, p.hidden_dim, positive),
            authority,
            search_end,
            admitted_bytes: bytes,
            state: Arc::new(Mutex::new(Counts::default())),
            portfolio_enabled: false,
            portfolio_phase_enabled: native == NativePolicy::Conditional,
            phase_all_enabled: false,
            phase_cube_enabled: false,
            prepaid: None,
            root_coupling: None,
        };
        budget.bank_capture(cuts, &capture)?;
        if native != NativePolicy::Disabled {
            let stage = NativeConstruction {
                source: budget.source,
                ledger: Arc::as_ptr(&budget.state),
                clocks: (budget.authority, budget.search_end),
            };
            let _ = native_publication::reserve_during_construction(cuts, &mut budget, stage);
        }
        Some(budget)
    }
    fn bank_capture(&self, cuts: &HullRootMipCuts<'_>, capture: &HullLpPrimalReport) -> Option<()> {
        if let Some(account) = &cuts.primal {
            if let Some(report) = &cuts.primal_report {
                report.lock().ok()?.bank_attempts += 1;
            }
            let mut account = account.lock().ok()?;
            if account.work() != capture.work_units || account.peak_bytes() != capture.peak_bytes {
                return None;
            }
            account.bank_once(
                cuts.problem,
                self.search_end,
                &self.state,
                &mut |work, payload| {
                    if work > MAX_WORK || payload != capture.peak_bytes {
                        return false;
                    }
                    let Ok(mut counts) = self.state.lock() else {
                        return false;
                    };
                    if counts.work != 0 || counts.reserved != 0 || counts.replay_attempts != 0 {
                        return false;
                    }
                    counts.work = work;
                    true
                },
            )?;
            if let Some(report) = &cuts.primal_report {
                let mut report = report.lock().ok()?;
                report.banked = true;
                report.stop_reason = "banked";
            }
        } else if capture.work_units != 0 || capture.peak_bytes != 0 {
            // A missing owner can never erase an already-spent capture debit.
            return None;
        }
        Some(())
    }
    pub(super) fn charge(&self, cost: usize) -> Option<()> {
        if !live(self.search_end) {
            self.stop("deadline");
            return None;
        }
        let mut c = self.state.lock().ok()?;
        let next = c.work.checked_add(cost)?;
        if next
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
        {
            c.reason = Some("work");
            return None;
        }
        c.work = next;
        Some(())
    }
    /// Native-cache work uses prepaid entry credit or leftover original credit.
    /// It never resets the root-search clock, quota or publication reserves.
    pub(super) fn child_charge(&self, cost: usize, call: bool, end: Instant) -> Option<()> {
        let mut c = self.state.lock().ok()?;
        if call {
            if c.child.attempted_calls >= child::MAX_CHILD_CALLS {
                c.child.stop_reason = "call-cap";
                return None;
            }
            c.child.attempted_calls += 1;
        } else {
            c.child.bind_attempts += 1;
        }
        // Consume eligible base-call or first-bind credit before any deadline
        // refusal. Clones share this ledger; admission already paid this work.
        let prepaid = if call {
            corners::take_base_call(&mut c.corner, cost)
        } else {
            std::mem::take(&mut c.prepaid_bind_work)
        };
        if !live(end.min(self.authority)) {
            c.child.stop_reason = "deadline";
            return None;
        }
        if prepaid != 0 {
            if cost > prepaid {
                c.child.stop_reason = if call {
                    "prepaid-call-refused"
                } else {
                    "prepaid-bind-refused"
                };
                return None;
            }
            c.child.stop_reason = "completed";
            return Some(());
        }
        let next = c.work.checked_add(cost)?;
        if next
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
        {
            c.child.stop_reason = "work";
            return None;
        }
        c.work = next;
        c.child.work_units = c.child.work_units.checked_add(cost)?;
        c.child.stop_reason = "completed";
        Some(())
    }
    pub(super) fn remaining(&self) -> Option<usize> {
        let c = self.state.lock().ok()?;
        MAX_WORK
            .checked_sub(c.work)?
            .checked_sub(c.reserved)?
            .checked_sub(c.component_reserved)
    }
    pub(super) fn can_check(&self) -> bool {
        self.state.lock().is_ok_and(|c| c.checks < MAX_CHECKS)
    }
    pub(super) fn payload_bytes(&self) -> Option<usize> {
        let c = self.state.lock().ok()?;
        self.admitted_bytes
            .checked_add(c.shared_bytes)?
            .checked_add(c.child.cache_bytes)?
            .checked_add(c.component_rows.admitted_bytes)
    }
    /// Only increases an unspent hold; never renews work, checks or replay slots.
    pub(super) fn reserve_at_least(&self, cost: usize) -> Option<()> {
        if !live(self.search_end) {
            return None;
        }
        let mut c = self.state.lock().ok()?;
        if c.replay_attempts != 0 {
            return None;
        }
        let next = c.reserved.max(cost.checked_mul(2)?);
        if c.work
            .checked_add(next)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
        {
            return None;
        }
        c.reserved = next;
        Some(())
    }
    /// Release only an excess UNSPENT hold after all larger alternatives died.
    /// Spent work, check slots and replay attempts are never refunded/reset.
    pub(super) fn trim_unspent_reservation(&self, cost: usize) -> Option<()> {
        let mut c = self.state.lock().ok()?;
        let needed = cost.checked_mul(2)?;
        if c.replay_attempts != 0 || needed > c.reserved {
            return None;
        }
        c.reserved = needed;
        Some(())
    }
    /// Quote simultaneous whole/shared buffers before the second plan exists.
    pub(super) fn admit_sharing(&self) -> Option<()> {
        if !live(self.search_end) {
            return None;
        }
        let (d, h, p) = self.dimensions;
        let bytes = p.checked_mul(256)?.checked_mul(192)?.checked_add(
            d.checked_mul(h)?
                .checked_add(h)?
                .checked_add(d)?
                .checked_add(1)?
                .checked_mul(128)?,
        )?;
        let mut c = self.state.lock().ok()?;
        if c.sharing_attempted {
            return None;
        }
        c.sharing_attempted = true;
        c.mass.attempted = true;
        c.mass.stop_reason = "proposal";
        if self
            .admitted_bytes
            .checked_add(bytes)?
            .checked_add(c.direction.cache_bytes)?
            .checked_add(c.child.cache_bytes)?
            .checked_add(c.component_rows.admitted_bytes)?
            > MAX_BYTES
        {
            c.mass.stop_reason = "payload";
            return None;
        }
        c.shared_bytes = bytes;
        Some(())
    }
    pub(super) fn reserve(&self, cost: usize) -> Option<()> {
        if !live(self.search_end) {
            return None;
        }
        let mut c = self.state.lock().ok()?;
        let reserved = cost.checked_mul(2)?;
        if c.reserved != 0
            || c.work
                .checked_add(reserved)?
                .checked_add(c.component_reserved)?
                > MAX_WORK
        {
            return None;
        }
        c.reserved = reserved;
        Some(())
    }
    pub(super) fn proof(&self, cost: usize, publication: bool) -> Option<()> {
        if !live(if publication {
            self.authority
        } else {
            self.search_end
        }) {
            self.stop("deadline");
            return None;
        }
        let mut c = self.state.lock().ok()?;
        if (publication && c.replay_attempts >= 2) || (!publication && c.checks >= MAX_CHECKS) {
            c.reason = Some("checks");
            return None;
        }
        let next = c.work.checked_add(cost)?;
        if publication && c.reserved < cost {
            c.reason = Some("work");
            return None;
        }
        let pub_hold = if publication {
            c.reserved - cost
        } else {
            c.reserved
        };
        if next
            .checked_add(pub_hold)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
        {
            c.reason = Some("work");
            return None;
        }
        c.work = next;
        if publication {
            c.replay_attempts += 1;
            c.reserved -= cost;
        } else {
            c.checks += 1;
        }
        Some(())
    }
    pub(super) fn stop(&self, reason: &'static str) {
        if let Ok(mut c) = self.state.lock() {
            if c.reason.is_none() {
                c.reason = Some(reason);
            }
        }
    }
}
