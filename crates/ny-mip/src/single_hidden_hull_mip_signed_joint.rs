// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Joint original-mask graph rows, paid after the old immutable result is banked.
//! Local pair rows never become a whole-q proof or sum overlapping signed mass.
use super::*;
#[path = "single_hidden_hull_mip_signed_joint_checked.rs"]
mod checked;
#[path = "single_hidden_hull_mip_signed_partial.rs"]
mod partial;
#[path = "single_hidden_hull_mip_signed_joint_resources.rs"]
mod resources;
#[path = "single_hidden_hull_mip_signed_joint_steering.rs"]
mod steering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedJointRowsReport {
    pub attempted: bool,
    pub prepaid_requested: bool,
    pub prepaid_admitted: bool,
    pub prepaid_work_units: usize,
    pub prepaid_bytes: usize,
    pub prepaid_used_work_units: usize,
    pub prepaid_checker_slots: usize,
    pub producer_attempts: usize,
    pub checks_before: usize,
    pub remaining_checks_before: usize,
    pub free_work_before: usize,
    pub eligible_pairs: usize,
    pub paid_pairs: usize,
    pub completed_pairs: usize,
    pub checker_slots: usize,
    pub quoted_masks: usize,
    pub checked_masks: usize,
    pub partial_eligible_pairs: usize,
    pub partial_paid_pairs: usize,
    pub partial_completed_pairs: usize,
    pub partial_selected_heads: usize,
    pub partial_tail_heads: usize,
    pub partial_checked_masks: usize,
    pub partial_tail_rechecks: usize,
    pub partial_work_units: usize,
    pub partial_prepared_rows: usize,
    pub partial_appended_rows: usize,
    pub coordinates: usize,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub prepared_rows: usize,
    pub prepared_head_terms: usize,
    pub prepared_terms: usize,
    pub appended_rows: usize,
    pub appended_terms: usize,
    pub stop_reason: &'static str,
}
impl Default for SignedJointRowsReport {
    fn default() -> Self {
        Self {
            attempted: false,
            prepaid_requested: false,
            prepaid_admitted: false,
            prepaid_work_units: 0,
            prepaid_bytes: 0,
            prepaid_used_work_units: 0,
            prepaid_checker_slots: 0,
            producer_attempts: 0,
            checks_before: 0,
            remaining_checks_before: 0,
            free_work_before: 0,
            eligible_pairs: 0,
            paid_pairs: 0,
            completed_pairs: 0,
            checker_slots: 0,
            quoted_masks: 0,
            checked_masks: 0,
            partial_eligible_pairs: 0,
            partial_paid_pairs: 0,
            partial_completed_pairs: 0,
            partial_selected_heads: 0,
            partial_tail_heads: 0,
            partial_checked_masks: 0,
            partial_tail_rechecks: 0,
            partial_work_units: 0,
            partial_prepared_rows: 0,
            partial_appended_rows: 0,
            coordinates: 0,
            work_units: 0,
            admitted_bytes: 0,
            prepared_rows: 0,
            prepared_head_terms: 0,
            prepared_terms: 0,
            appended_rows: 0,
            appended_terms: 0,
            stop_reason: "not-attempted",
        }
    }
}
fn stop(b: &Budget, reason: &'static str) {
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.joint.stop_reason = reason;
    }
}
fn start(row: &SignedMajorant) -> bool {
    let Ok(mut c) = row.budget.state.lock() else {
        return false;
    };
    if c.replay_attempts != 1
        || c.source_replays != 1
        || !c.component_rows.admitted
        || c.component_rows.joint.attempted
    {
        return false;
    }
    let checks = c.checks;
    let free = MAX_WORK
        .checked_sub(c.work)
        .and_then(|v| v.checked_sub(c.reserved))
        .and_then(|v| v.checked_sub(c.component_reserved))
        .unwrap_or(0);
    let r = &mut c.component_rows.joint;
    r.attempted = true;
    r.producer_attempts = 1;
    r.checks_before = checks;
    r.remaining_checks_before = MAX_CHECKS.saturating_sub(checks);
    r.free_work_before = free;
    r.stop_reason = "setup";
    if checks >= MAX_CHECKS && row.budget.prepaid.is_none() {
        r.stop_reason = "check-slots";
        return false;
    }
    true
}
/// Allocation-free comparison to the already banked immutable full witness.
/// This new scan is included in the setup quote, not performed before payment.
fn witness_matches(row: &SignedMajorant) -> bool {
    let mut bits = row.witness.iter().copied();
    let p = &row.plan;
    if bits.next() != Some(p.components.len() as u64)
        || bits.next() != Some(p.leftover.len() as u64)
    {
        return false;
    }
    for &j in &p.leftover {
        if bits.next() != Some(j as u64) {
            return false;
        }
    }
    for c in &p.components {
        for len in [
            c.negative.len(),
            c.positive.len(),
            c.positive_units.len(),
            c.g.len(),
            c.alpha.len(),
        ] {
            if bits.next() != Some(len as u64) {
                return false;
            }
        }
        for &j in c.negative.iter().chain(&c.positive) {
            if bits.next() != Some(j as u64) {
                return false;
            }
        }
        for &u in &c.positive_units {
            if bits.next() != Some(u64::from(u)) {
                return false;
            }
        }
        for &v in c.g.iter().chain(&c.alpha) {
            if bits.next() != Some(v.to_bits()) {
                return false;
            }
        }
    }
    bits.next().is_none()
}
/// Paid metadata-only feasibility, used before any union feature/witness scan.
pub(super) fn can_fund(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    free: usize,
    used_bytes: usize,
) -> bool {
    let Some((setup, bytes)) = resources::setup(row, cuts) else {
        return false;
    };
    for (i, a) in row.plan.components.iter().enumerate() {
        for b in row.plan.components.iter().skip(i + 1) {
            let q = if a.negative.len().saturating_add(b.negative.len()) > 8 {
                partial::quote(cuts.problem, a, b)
            } else {
                resources::quote(cuts.problem, a, b)
            };
            let Some((_, work, payload)) = q else {
                continue;
            };
            if setup
                .checked_add(work)
                .is_some_and(|v| v <= free && v <= resources::MAX_JOINT_WORK)
                && bytes
                    .checked_add(payload)
                    .and_then(|v| used_bytes.checked_add(v))
                    .is_some_and(|v| v <= MAX_BYTES)
            {
                return true;
            }
        }
    }
    false
}
fn union_rows(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Vec<RowSpec> {
    let Some((work, bytes)) = resources::setup(row, cuts) else {
        stop(&row.budget, "shape");
        return Vec::new();
    };
    if resources::pay(row, work, bytes, end).is_none() {
        return Vec::new();
    }
    if !witness_matches(row) || !signature(cuts).eq(row.signature.iter().copied()) {
        stop(&row.budget, "source-or-witness");
        return Vec::new();
    }
    let Some(checked) = checked::produce(row, cuts, end, poll) else {
        stop(&row.budget, "incomplete-joint-producer");
        return Vec::new();
    };
    checked::export(row, cuts, parts, binding, &checked, end, poll).unwrap_or_else(|| {
        stop(&row.budget, "binding-or-export");
        Vec::new()
    })
}
/// Both optional families share THIS one stored absolute 50ms window. A later
/// append cannot renew it because start() irreversibly owns the attempt.
pub(super) fn prepare(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Vec<RowSpec> {
    if !binding.matches(cuts, parts) || row.source != std::ptr::from_ref(cuts.problem) as usize {
        return Vec::new();
    }
    if row.budget.root_coupling.is_some() {
        stop(&row.budget, "exclusive-root-proposal");
        return Vec::new();
    }
    if !start(row) {
        return Vec::new();
    }
    let Some(window) = Instant::now().checked_add(Duration::from_millis(50)) else {
        return Vec::new();
    };
    let end = end
        .min(row.authority())
        .min(row.budget.search_end)
        .min(cuts.deadline)
        .min(window);
    if let Some(ticket) = &row.budget.prepaid {
        // New policy is exclusive: one already-funded union, no domination
        // attempt and no second optional family if it refuses or expires.
        if !ticket.begin(row, end) {
            stop(&row.budget, "prepaid-owner-or-clock");
            return Vec::new();
        }
        if require_hull_environment().is_err() || !live(end) || !poll() {
            stop(&row.budget, "deadline-or-poll");
            return Vec::new();
        }
        return union_rows(row, cuts, parts, binding, end, poll);
    }
    if require_hull_environment().is_err() || !live(end) || !poll() {
        stop(&row.budget, "deadline-or-poll");
        return Vec::new();
    }
    let mut rows = domination::prepare(row, cuts, parts, binding, end, poll);
    if let Ok(mut c) = row.budget.state.lock() {
        let checks = c.checks;
        let free = MAX_WORK
            .checked_sub(c.work)
            .and_then(|v| v.checked_sub(c.reserved))
            .and_then(|v| v.checked_sub(c.component_reserved))
            .unwrap_or(0);
        c.component_rows.joint.checks_before = checks;
        c.component_rows.joint.remaining_checks_before = MAX_CHECKS.saturating_sub(checks);
        c.component_rows.joint.free_work_before = free;
    }
    if !domination::allow_joint(row, cuts) {
        stop(&row.budget, "domination-retained-no-funded-pair");
        return rows;
    }
    rows.extend(union_rows(row, cuts, parts, binding, end, poll));
    rows
}
pub(super) fn reset(b: &Budget) {
    if let Ok(mut c) = b.state.lock() {
        let r = &mut c.component_rows.joint;
        r.partial_prepared_rows = 0;
        r.prepared_rows = 0;
        r.prepared_head_terms = 0;
        r.prepared_terms = 0;
    }
}
pub(super) fn commit(c: &mut Counts) {
    let r = &mut c.component_rows.joint;
    r.partial_appended_rows = r.partial_prepared_rows;
    r.appended_rows = r.prepared_rows;
    r.appended_terms = r.prepared_terms;
    if r.appended_rows > 0 {
        r.stop_reason = "appended";
    }
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_joint_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_joint_test_fixture.rs"]
mod test_fixture;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_joint_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_joint_prepaid_tests.rs"]
mod prepaid_tests;
