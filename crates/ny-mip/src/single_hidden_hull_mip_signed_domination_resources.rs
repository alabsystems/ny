// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Whole optional quote paid before metadata, advice, or original support.
use super::*;
pub(super) const MAX_DOMINATION_WORK: usize = 664_696;
pub(super) const MAX_TARGETS: usize = 32;
pub(super) struct Ticket {
    pub end: Instant,
    pub targets: usize,
    pub sparse: usize,
    ledger: Arc<std::sync::Mutex<Counts>>,
    accepted_end: Instant,
    dense: Option<dense::Quote>,
    group: Option<group::Quote>,
    branch: Option<branch::Quote>,
    route: std::sync::atomic::AtomicU8,
    branch_nominal_used: std::sync::atomic::AtomicBool,
    group_advice_used: std::sync::atomic::AtomicBool,
    group_support_used: std::sync::atomic::AtomicBool,
}
impl Ticket {
    pub(super) fn dense_quote(&self) -> Option<&dense::Quote> {
        self.dense
            .as_ref()
            .or_else(|| self.group.as_ref().map(group::Quote::base))
            .or_else(|| self.branch.as_ref().map(branch::Quote::base))
    }
    pub(super) fn group_quote(&self) -> Option<&group::Quote> {
        self.group.as_ref()
    }
    pub(super) fn branch_quote(&self) -> Option<&branch::Quote> {
        self.branch.as_ref()
    }
    fn pair_mode(&self) -> bool {
        self.group.is_some() || self.branch.is_some()
    }
    pub(super) fn alias_limit(&self) -> usize {
        self.dense_quote().map_or(0, |q| q.negative_heads)
    }
    pub(super) fn support_limit(&self) -> usize {
        self.dense_quote().map_or(4, |q| q.positive_limit)
    }
}
pub(super) fn quote(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    targets: usize,
    sparse: usize,
) -> Option<(usize, usize)> {
    let p = cuts.problem;
    let h = p.hidden_dim;
    let width = p.input_dim.checked_add(1)?;
    let n = cuts.planes.len();
    let positive = row.budget.dimensions.2;
    let components = row.plan.components.len();
    if targets == 0 || targets > n.min(MAX_TARGETS) || sparse > targets.min(2) {
        return None;
    }
    let setup = h
        .checked_mul(width)?
        .checked_mul(64)?
        .checked_add(n.checked_mul(positive)?.checked_mul(16)?)?
        .checked_add(h.checked_add(1)?.checked_mul(128)?)?
        .checked_add(8192)?
        .checked_add(components.checked_mul(components)?.checked_mul(32)?)?
        .checked_add(components.checked_mul(128)?)?;
    let scores = targets
        .checked_mul(4)?
        .checked_add(sparse.checked_mul(16)?)?;
    let work = setup
        .checked_add(width.checked_mul(scores)?.checked_mul(64)?)?
        .checked_add(width.checked_mul(targets)?.checked_mul(5 * 128)?)?;
    let bytes = h
        .checked_mul(128)?
        .checked_add(
            targets
                .checked_mul(width.checked_add(16)?)?
                .checked_mul(128)?,
        )?
        .checked_add(width.checked_mul(64)?)?
        .checked_add(32768)?;
    Some((work, bytes))
}
fn bytes(row: &SignedMajorant, c: &Counts) -> Option<usize> {
    row.budget
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.child.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)
}
pub(super) fn free(c: &Counts) -> Option<usize> {
    MAX_WORK
        .checked_sub(c.work)?
        .checked_sub(c.reserved)?
        .checked_sub(c.component_reserved)
}
pub(super) fn admit(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
) -> Option<Ticket> {
    let account = cuts.primal.as_ref().and_then(|a| a.lock().ok());
    admit_account(row, cuts, end, account.as_deref(), Selection::Sparse)
}
pub(super) fn admit_point(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    account: Option<&primal_advice::Captured<'_>>,
) -> Option<Ticket> {
    if account.is_none() && cuts.primal.is_none() {
        return admit(row, cuts, end);
    }
    admit_account(row, cuts, end, account, Selection::Automatic)
}
pub(super) fn admit_group_point(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    account: Option<&primal_advice::Captured<'_>>,
) -> Option<Ticket> {
    admit_account(row, cuts, end, account, Selection::Group)
}
pub(super) fn admit_branch_point(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    account: Option<&primal_advice::Captured<'_>>,
) -> Option<Ticket> {
    admit_account(row, cuts, end, account, Selection::Branch)
}
#[derive(Clone, Copy)]
enum Selection {
    Sparse,
    Automatic,
    Group,
    Branch,
}
fn admit_account(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    account: Option<&primal_advice::Captured<'_>>,
    selection: Selection,
) -> Option<Ticket> {
    let b = &row.budget;
    let p = cuts.problem;
    if row.source != std::ptr::from_ref(p) as usize
        || b.source != row.source
        || b.dimensions.0 != p.input_dim
        || b.dimensions.1 != p.hidden_dim
        || !capacity(p)
    {
        return None;
    }
    let end = end.min(b.search_end).min(b.authority).min(cuts.deadline);
    let capture_work = if let Some(a) = account {
        a.debt_for(p, end, &b.state)?
    } else {
        if cuts.primal.is_some() {
            return None;
        }
        0
    };
    let point = if matches!(
        selection,
        Selection::Automatic | Selection::Group | Selection::Branch
    ) {
        account.and_then(|a| a.point(p, end, &b.state))
    } else {
        None
    };
    let mut c = b.state.lock().ok()?;
    if c.replay_attempts != 1
        || c.source_replays != 1
        || !c.component_rows.admitted
        || c.component_rows.domination.attempted
    {
        return None;
    }
    let free = free(&c)?;
    let checks = c.checks;
    let r = &mut c.component_rows.domination;
    r.attempted = true;
    r.producer_attempts = 1;
    r.free_work_before = free;
    r.checks_before = checks;
    r.stop_reason = "quote";
    if !live(end) || checks >= MAX_CHECKS || b.dimensions.2 == 0 {
        r.stop_reason = "deadline-shape-or-checks";
        return None;
    }
    let available_bytes = MAX_BYTES.checked_sub(bytes(row, &c)?)?;
    let chosen = choose(
        row,
        cuts,
        free,
        available_bytes,
        capture_work,
        point,
        selection,
    );
    let Some(Choice {
        targets,
        sparse,
        work,
        payload,
        dense,
        group,
        branch,
    }) = chosen
    else {
        c.component_rows.domination.stop_reason = "work-or-payload";
        return None;
    };
    // The fixed-size quote preflight belongs to the existing selection escrow.
    // No debit, optional payload, or checker grant precedes this consistency gate.
    let total = capture_work.checked_add(work)?;
    if total > MAX_DOMINATION_WORK
        || dense
            .as_ref()
            .is_some_and(|q| q.checked_total(capture_work) != Some(total))
        || group.as_ref().is_some_and(|q| {
            q.checked_total(p, row.plan.components.len(), capture_work) != Some(total)
        })
        || branch.as_ref().is_some_and(|q| {
            q.checked_total(p, row.plan.components.len(), capture_work) != Some(total)
        })
    {
        return None;
    }
    let next = c.work.checked_add(work)?;
    let component_work = c.component_rows.work_units.checked_add(work)?;
    let component_bytes = c.component_rows.admitted_bytes.checked_add(payload)?;
    c.work = next;
    c.checks += 1;
    c.component_rows.work_units = component_work;
    c.component_rows.admitted_bytes = component_bytes;
    let r = &mut c.component_rows.domination;
    r.admitted = true;
    r.funded_targets = targets;
    r.funded_sparse_fits = sparse;
    r.dense = dense.is_some() || group.is_some() || branch.is_some();
    r.group_allocation = group.is_some() || branch.is_some();
    r.branch_tau_mode = branch.is_some();
    r.group_nominal_work = group.as_ref().map_or(0, |q| q.nominal_work());
    r.group_checker_tail = group.as_ref().map_or(0, |q| q.checker_tail());
    if let Some(q) = branch.as_ref() {
        r.group_nominal_work = q.nominal;
        r.group_checker_tail = q.checker;
        r.tail_single_work = q.single;
        r.tail_branch_work = q.branch;
        r.branch_tau_optimizer_work = q.optimizer;
    }
    let base = dense
        .as_ref()
        .or_else(|| group.as_ref().map(group::Quote::base))
        .or_else(|| branch.as_ref().map(branch::Quote::base));
    r.funded_positive_limit = base.map_or(4, |q| q.positive_limit);
    r.pool_heads = base.map_or(0, |q| q.pool_heads);
    r.funded_alias_heads = base.map_or(0, |q| q.negative_heads);
    r.alias_work_units = base.map_or(0, |q| q.alias_work);
    r.alias_bytes = base.map_or(0, |q| q.alias_bytes);
    r.capture_work_units = capture_work;
    r.total_optional_work_units = total;
    r.work_units = work;
    r.admitted_bytes = payload;
    r.checker_slots = 1;
    r.stop_reason = "paid";
    Some(Ticket {
        end,
        targets,
        sparse,
        ledger: Arc::clone(&b.state),
        accepted_end: end,
        dense,
        group,
        branch,
        route: std::sync::atomic::AtomicU8::new(0),
        branch_nominal_used: std::sync::atomic::AtomicBool::new(false),
        group_advice_used: std::sync::atomic::AtomicBool::new(false),
        group_support_used: std::sync::atomic::AtomicBool::new(false),
    })
}
pub(super) fn support_attempt(row: &SignedMajorant, t: &Ticket, end: Instant) -> Option<Instant> {
    support_attempt_inner(row, t, end, false)
}
pub(super) fn branch_support_attempt(
    row: &SignedMajorant,
    t: &Ticket,
    end: Instant,
) -> Option<Instant> {
    support_attempt_inner(row, t, end, true)
}
fn support_attempt_inner(
    row: &SignedMajorant,
    t: &Ticket,
    end: Instant,
    branch_request: bool,
) -> Option<Instant> {
    let route = t.route.load(std::sync::atomic::Ordering::Relaxed);
    if (t.branch.is_some() && route != if branch_request { 2 } else { 1 })
        || (branch_request && t.branch.is_none())
    {
        return None;
    }
    if !Arc::ptr_eq(&t.ledger, &row.budget.state) || t.end != t.accepted_end {
        return None;
    }
    let end = end
        .min(t.end)
        .min(row.authority())
        .min(row.budget.search_end);
    let mut c = row.budget.state.lock().ok()?;
    if c.replay_attempts != 1 || c.source_replays != 1 {
        return None;
    }
    let r = &mut c.component_rows.domination;
    if !r.admitted
        || t.targets != r.funded_targets
        || t.sparse != r.funded_sparse_fits
        || t.support_limit() != r.funded_positive_limit
        || t.alias_limit() != r.funded_alias_heads
        || t.pair_mode() != r.group_allocation
        || r.support_attempts >= r.funded_targets
    {
        return None;
    }
    // The single OR pair winner consumes this one immutable capability.
    if t.pair_mode()
        && (t.targets != 1
            || t.sparse != 0
            || t.group_support_used
                .swap(true, std::sync::atomic::Ordering::Relaxed))
    {
        return None;
    }
    r.support_attempts += 1;
    if !live(end) {
        r.stop_reason = "deadline";
        return None;
    }
    Some(end)
}
/// This metadata walk is included in the domination quote. No feature or
/// original-alpha scan is needed to reject an unfundable second family.
pub(super) fn joint_can_fund(row: &SignedMajorant, cuts: &HullRootMipCuts<'_>) -> bool {
    let Ok(c) = row.budget.state.lock() else {
        return false;
    };
    let Some(free) = free(&c) else { return false };
    let Some(used) = bytes(row, &c) else {
        return false;
    };
    c.checks < MAX_CHECKS && joint::can_fund(row, cuts, free, used)
}
/// The nominal proposal budget is once-only too; no repeated paid-prefix advice.
pub(super) fn begin_advice(row: &SignedMajorant, t: &Ticket) -> Option<()> {
    if !Arc::ptr_eq(&t.ledger, &row.budget.state) || t.end != t.accepted_end {
        return None;
    }
    let mut c = row.budget.state.lock().ok()?;
    if c.replay_attempts != 1 || c.source_replays != 1 {
        return None;
    }
    let r = &mut c.component_rows.domination;
    if !r.admitted
        || r.advice_attempts != 0
        || t.targets != r.funded_targets
        || t.sparse != r.funded_sparse_fits
        || t.support_limit() != r.funded_positive_limit
        || t.alias_limit() != r.funded_alias_heads
        || t.pair_mode() != r.group_allocation
        || !live(t.end)
    {
        return None;
    }
    if t.pair_mode()
        && (t.targets != 1
            || t.sparse != 0
            || t.group_advice_used
                .swap(true, std::sync::atomic::Ordering::Relaxed))
    {
        return None;
    }
    r.advice_attempts = 1;
    Some(())
}

struct Choice {
    targets: usize,
    sparse: usize,
    work: usize,
    payload: usize,
    dense: Option<dense::Quote>,
    group: Option<group::Quote>,
    branch: Option<branch::Quote>,
}
fn choose(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    free: usize,
    bytes: usize,
    capture_work: usize,
    point: Option<&primal_advice::Point>,
    selection: Selection,
) -> Option<Choice> {
    let cap = MAX_DOMINATION_WORK.checked_sub(capture_work)?;
    let n = cuts.planes.len();
    let all = n.min(MAX_TARGETS);
    let sparse = if row.budget.dimensions.2 > 1 {
        all.min(2)
    } else {
        0
    };
    // Generic complete-schedule pressure, never an asset, model name or outcome.
    let affordable_all = n <= MAX_TARGETS
        && quote(row, cuts, all, sparse).is_some_and(|(w, p)| w <= free.min(cap) && p <= bytes);
    if !affordable_all {
        if let Some(point) = point.filter(|p| p.positive.len() > 1) {
            // Only a fully admitted captured point enters its prepaid8192
            // selection escrow. No new planning on sparse/no-point refusal.
            let branch_plan = if matches!(selection, Selection::Branch) {
                branch::Planning::new(cuts.problem)
            } else {
                None
            };
            let mut canonical_fallback = None;
            for q in (1..=32).rev() {
                if let Some(dense) = dense::quote_with_aliases(
                    cuts.problem,
                    row.plan.components.len(),
                    capture_work,
                    point.positive.len(),
                    point.negative.len(),
                    q,
                ) {
                    if dense.incremental_work <= free && dense.bytes <= bytes {
                        if matches!(selection, Selection::Group | Selection::Branch)
                            && point.negative.len() >= 2
                        {
                            // One bounded32-quote loop,one scalar-only base clone.
                            if canonical_fallback.is_none() {
                                canonical_fallback = Some(Choice {
                                    targets: 1,
                                    sparse: 0,
                                    work: dense.incremental_work,
                                    payload: dense.bytes,
                                    dense: Some(dense.clone()),
                                    group: None,
                                    branch: None,
                                });
                            }
                            if let Some(plan) = branch_plan.as_ref() {
                                if let Some(branch) = plan.quote_base(
                                    cuts.problem,
                                    row.plan.components.len(),
                                    capture_work,
                                    dense,
                                ) {
                                    if branch.incremental_work() <= free && branch.bytes() <= bytes
                                    {
                                        return Some(Choice {
                                            targets: 1,
                                            sparse: 0,
                                            work: branch.incremental_work(),
                                            payload: branch.bytes(),
                                            dense: None,
                                            group: None,
                                            branch: Some(branch),
                                        });
                                    }
                                }
                                continue;
                            }
                            if let Some(group) = group::quote(
                                cuts.problem,
                                row.plan.components.len(),
                                capture_work,
                                dense,
                            ) {
                                if group.incremental_work() <= free && group.bytes() <= bytes {
                                    return Some(Choice {
                                        targets: 1,
                                        sparse: 0,
                                        work: group.incremental_work(),
                                        payload: group.bytes(),
                                        dense: None,
                                        group: Some(group),
                                        branch: None,
                                    });
                                }
                            }
                        } else {
                            return Some(Choice {
                                targets: 1,
                                sparse: 0,
                                work: dense.incremental_work,
                                payload: dense.bytes,
                                dense: Some(dense),
                                group: None,
                                branch: None,
                            });
                        }
                    }
                }
            }
            if canonical_fallback.is_some() {
                return canonical_fallback;
            }
        }
    }
    for targets in (1..=all).rev() {
        let sparse = if row.budget.dimensions.2 > 1 {
            targets.min(2)
        } else {
            0
        };
        if let Some((work, payload)) = quote(row, cuts, targets, sparse) {
            if work <= free.min(cap) && payload <= bytes {
                return Some(Choice {
                    targets,
                    sparse,
                    work,
                    payload,
                    dense: None,
                    group: None,
                    branch: None,
                });
            }
        }
    }
    None
}

/// Latch once, BEFORE either optional tail. Impossible pairs preserve gradient.
pub(super) fn select_branch_route(
    row: &SignedMajorant,
    t: &Ticket,
    seed_useful: bool,
    pair_eligible: bool,
) -> Option<bool> {
    if t.branch.is_none()
        || !Arc::ptr_eq(&t.ledger, &row.budget.state)
        || t.end != t.accepted_end
        || !live(t.end)
    {
        return None;
    }
    let single = seed_useful || !pair_eligible;
    t.route
        .compare_exchange(
            0,
            if single { 1 } else { 2 },
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .ok()?;
    let mut c = row.budget.state.lock().ok()?;
    if c.replay_attempts != 1
        || c.source_replays != 1
        || c.component_rows.domination.advice_attempts != 1
    {
        return None;
    }
    let r = &mut c.component_rows.domination;
    r.exclusive_seed_useful = Some(seed_useful);
    r.gradient_skipped = !single;
    Some(single)
}
pub(super) fn begin_branch_tail(row: &SignedMajorant, t: &Ticket) -> bool {
    if t.branch.is_none()
        || t.route.load(std::sync::atomic::Ordering::Relaxed) != 2
        || !Arc::ptr_eq(&t.ledger, &row.budget.state)
        || t.end != t.accepted_end
        || !live(t.end)
        || t.branch_nominal_used
            .swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        return false;
    }
    let Ok(mut c) = row.budget.state.lock() else {
        return false;
    };
    if c.replay_attempts != 1 || c.source_replays != 1 {
        return false;
    }
    let r = &mut c.component_rows.domination;
    if !r.admitted || !r.branch_tau_mode || r.advice_attempts != 1 {
        return false;
    }
    r.group_nominal_attempts = 1;
    r.branch_tau_attempts = 1;
    true
}
