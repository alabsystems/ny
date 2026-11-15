// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Sparse original-graph relations. They never become a whole-q proof token.
use super::*;
#[path = "single_hidden_hull_mip_signed_domination_alias.rs"]
mod alias;
#[path = "single_hidden_hull_mip_signed_domination_branch.rs"]
mod branch;
#[path = "single_hidden_hull_mip_signed_domination_chain.rs"]
mod chain;
#[path = "single_hidden_hull_mip_signed_domination_dense.rs"]
mod dense;
#[path = "single_hidden_hull_mip_signed_domination_group.rs"]
mod group;
#[path = "single_hidden_hull_mip_signed_domination_kernel.rs"]
mod kernel;
#[path = "single_hidden_hull_mip_signed_domination_priced.rs"]
mod priced;
#[path = "single_hidden_hull_mip_signed_domination_resources.rs"]
mod resources;
#[path = "single_hidden_hull_mip_signed_domination_steering.rs"]
mod steering;
#[derive(Debug, Clone, PartialEq)]
pub struct SignedDominationReport {
    pub attempted: bool,
    pub admitted: bool,
    pub producer_attempts: usize,
    pub advice_attempts: usize,
    pub export_attempts: usize,
    pub free_work_before: usize,
    pub checks_before: usize,
    pub checker_slots: usize,
    pub funded_targets: usize,
    pub funded_sparse_fits: usize,
    pub dense: bool,
    pub group_allocation: bool,
    pub branch_tau_mode: bool,
    pub exclusive_seed_useful: Option<bool>,
    pub gradient_skipped: bool,
    pub tail_single_work: usize,
    pub tail_branch_work: usize,
    pub branch_tau_attempts: usize,
    pub branch_tau_completed: usize,
    pub branch_tau_mask_prices: Option<[f64; 4]>,
    pub branch_tau_values: Option<[f64; 4]>,
    pub branch_tau_selected: bool,
    pub branch_tau_checked_masks: usize,
    pub branch_tau_optimizer_work: usize,

    pub group_nominal_attempts: usize,
    pub group_nominal_work: usize,
    pub group_checker_tail: usize,
    pub group_nominal_masks: usize,
    pub group_checked_masks: usize,
    pub group_anchor: Option<usize>,
    pub group_partner: Option<usize>,
    pub group_nominal_lhs: Option<f64>,
    pub group_nominal_bound: Option<f64>,
    pub priced_step_work_units: usize,
    pub priced_sort_work_units: usize,
    pub priced_step_bytes: usize,
    pub priced_step_attempted: bool,
    pub priced_step_completed: bool,
    pub priced_step_accepted: bool,
    pub funded_positive_limit: usize,
    pub pool_heads: usize,
    pub funded_alias_heads: usize,
    pub alias_work_units: usize,
    pub alias_bytes: usize,
    pub checked_alias_rows: usize,
    pub prepared_alias_rows: usize,
    pub appended_alias_rows: usize,
    pub capture_work_units: usize,
    pub total_optional_work_units: usize,
    pub sparse_fits: usize,
    pub nominal_scores: usize,
    pub support_attempts: usize,
    pub completed_supports: usize,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub prepared_rows: usize,
    pub prepared_head_terms: usize,
    pub prepared_terms: usize,
    pub appended_rows: usize,
    pub appended_terms: usize,
    pub joint_skipped_unfunded: bool,
    pub stop_reason: &'static str,
}
impl Default for SignedDominationReport {
    fn default() -> Self {
        Self {
            attempted: false,
            admitted: false,
            producer_attempts: 0,
            advice_attempts: 0,
            export_attempts: 0,
            free_work_before: 0,
            checks_before: 0,
            checker_slots: 0,
            funded_targets: 0,
            funded_sparse_fits: 0,
            dense: false,
            group_allocation: false,
            branch_tau_mode: false,
            exclusive_seed_useful: None,
            gradient_skipped: false,
            tail_single_work: 0,
            tail_branch_work: 0,
            branch_tau_attempts: 0,
            branch_tau_completed: 0,
            branch_tau_mask_prices: None,
            branch_tau_values: None,
            branch_tau_selected: false,
            branch_tau_checked_masks: 0,
            branch_tau_optimizer_work: 0,

            group_nominal_attempts: 0,
            group_nominal_work: 0,
            group_checker_tail: 0,
            group_nominal_masks: 0,
            group_checked_masks: 0,
            group_anchor: None,
            group_partner: None,
            group_nominal_lhs: None,
            group_nominal_bound: None,
            priced_step_work_units: 0,
            priced_sort_work_units: 0,
            priced_step_bytes: 0,
            priced_step_attempted: false,
            priced_step_completed: false,
            priced_step_accepted: false,
            funded_positive_limit: 4,
            pool_heads: 0,
            funded_alias_heads: 0,
            alias_work_units: 0,
            alias_bytes: 0,
            checked_alias_rows: 0,
            prepared_alias_rows: 0,
            appended_alias_rows: 0,
            capture_work_units: 0,
            total_optional_work_units: 0,
            sparse_fits: 0,
            nominal_scores: 0,
            support_attempts: 0,
            completed_supports: 0,
            work_units: 0,
            admitted_bytes: 0,
            prepared_rows: 0,
            prepared_head_terms: 0,
            prepared_terms: 0,
            appended_rows: 0,
            appended_terms: 0,
            joint_skipped_unfunded: false,
            stop_reason: "not-attempted",
        }
    }
}
fn stop(b: &Budget, reason: &'static str) {
    if let Ok(mut c) = b.state.lock() {
        c.component_rows.domination.stop_reason = reason;
    }
}
/// Only the fully authenticated first-publication callback enters here.
/// Source/old tuple and original model have ALREADY been replayed/compared.
/// This graph proof uses neither old alpha nor old component G: it freshly
/// prices only original preactivations and the new point lambda/v witness.
pub(super) fn prepare(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Vec<RowSpec> {
    let mut cancelled = false;
    let mut observed_poll = || {
        if cancelled {
            return false;
        }
        let value = poll();
        cancelled = !value;
        value
    };
    let poll: &mut dyn FnMut() -> bool = &mut observed_poll;
    if !binding.matches(cuts, parts) || row.source != std::ptr::from_ref(cuts.problem) as usize {
        return Vec::new();
    }
    let account = cuts.primal.as_ref().and_then(|a| a.lock().ok());
    let point = account
        .as_ref()
        .and_then(|a| a.point(cuts.problem, end, &row.budget.state));
    let admitted = if cuts.branch_tau_mode {
        resources::admit_branch_point(row, cuts, end, account.as_deref())
    } else if cuts.group_allocation_mode {
        resources::admit_group_point(row, cuts, end, account.as_deref())
    } else {
        resources::admit_point(row, cuts, end, account.as_deref())
    };
    let Some(ticket) = admitted else {
        return Vec::new();
    };
    if require_hull_environment().is_err() || !live(ticket.end) || !poll() {
        stop(&row.budget, "deadline-fp-or-poll");
        return Vec::new();
    }
    let proposals = if let Some(paid) = ticket.dense_quote() {
        if resources::begin_advice(row, &ticket).is_none() {
            return Vec::new();
        }
        let Some(point) = point else {
            return Vec::new();
        };
        let advice = if ticket.branch_quote().is_some() {
            dense::exclusive_advice(row, cuts, point, paid, &ticket, ticket.end, poll)
        } else {
            dense::advice(row, cuts, point, paid, ticket.end, poll)
        };
        let Some(advice) = advice else {
            stop(&row.budget, "incomplete-dense-advice");
            return Vec::new();
        };
        if advice.useful {
            vec![advice.single]
        } else if ticket.branch_quote().is_some() {
            let Some(proposal) =
                branch::proposal(row, cuts, point, &ticket, &advice, ticket.end, poll)
            else {
                stop(&row.budget, "no-positive-branch-tau-violation");
                return Vec::new();
            };
            let Some(token) =
                group::kernel::branch::check(row, cuts, &ticket, proposal, ticket.end, poll)
            else {
                stop(&row.budget, "incomplete-branch-tau-support");
                return Vec::new();
            };
            return group::kernel::export(row, cuts, parts, binding, &token, ticket.end, poll)
                .unwrap_or_default();
        } else if let Some(group_quote) = ticket.group_quote() {
            let Some(proposal) =
                group::proposal(row, cuts, point, group_quote, &advice, ticket.end, poll)
            else {
                stop(&row.budget, "no-positive-group-violation");
                return Vec::new();
            };
            let Some(token) = group::kernel::check(row, cuts, &ticket, proposal, ticket.end, poll)
            else {
                stop(&row.budget, "incomplete-group-support");
                return Vec::new();
            };
            return group::kernel::export(row, cuts, parts, binding, &token, ticket.end, poll)
                .unwrap_or_default();
        } else {
            stop(&row.budget, "no-positive-dense-violation");
            return Vec::new();
        }
    } else {
        let Some(proposals) = steering::proposals(row, cuts, &ticket, poll) else {
            stop(&row.budget, "incomplete-advice");
            return Vec::new();
        };
        proposals
    };
    if proposals.is_empty() {
        stop(&row.budget, "no-positive-fit");
        return Vec::new();
    }
    let mut checked = Vec::with_capacity(proposals.len());
    for proposal in proposals {
        let token = if ticket.dense_quote().is_some() {
            point.and_then(|point| {
                kernel::check_with_aliases(
                    row,
                    cuts,
                    &ticket,
                    proposal,
                    &point.negative,
                    ticket.end,
                    poll,
                )
            })
        } else {
            kernel::check(row, cuts, &ticket, proposal, ticket.end, poll)
        };
        let Some(token) = token else {
            stop(&row.budget, "incomplete-original-support");
            return Vec::new();
        };
        checked.push(token);
    }
    let Some(rows) = kernel::export(row, cuts, parts, binding, &checked, ticket.end, poll) else {
        stop(&row.budget, "binding-or-export-refusal");
        return Vec::new();
    };
    rows
}
/// A paid length-only feasibility check precedes any old union setup charge.
/// When there is ample credit BOTH families retain their original behavior.
pub(super) fn allow_joint(row: &SignedMajorant, cuts: &HullRootMipCuts<'_>) -> bool {
    let paid = row
        .budget
        .state
        .lock()
        .is_ok_and(|c| c.component_rows.domination.admitted);
    if !paid || resources::joint_can_fund(row, cuts) {
        return true;
    }
    if let Ok(mut c) = row.budget.state.lock() {
        c.component_rows.domination.joint_skipped_unfunded = true;
    }
    false
}
pub(super) fn reset(b: &Budget) {
    if let Ok(mut c) = b.state.lock() {
        let r = &mut c.component_rows.domination;
        r.prepared_rows = 0;
        r.prepared_head_terms = 0;
        r.prepared_terms = 0;
        r.prepared_alias_rows = 0;
    }
}
pub(super) fn commit(c: &mut Counts) {
    let r = &mut c.component_rows.domination;
    r.appended_rows = r.prepared_rows;
    r.appended_terms = r.prepared_terms;
    r.appended_alias_rows = r.prepared_alias_rows;
    if r.appended_rows > 0 {
        r.stop_reason = "appended";
    }
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_test_fixture.rs"]
mod test_fixture;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_tests.rs"]
mod tests;
