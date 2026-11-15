// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optional five-head extension of completed small negative-group rows.
//! Search and the old row's coefficients propose represented masses only.
//! A new row exists only after ALL32 full-source affine mask supports finish.
//! This module never imports the old RHS, an objective, or an AY status.

use super::*;

const HEADS: usize = 5;
const MAX_GROWN: usize = 4;
const MAX_POOL: usize = 16;
const MAX_BASES: usize = 4;
const MAX_RANK_TRIALS: usize = MAX_BASES * MAX_POOL;
const SHARED_MASKS: usize = 256;
const SHARED_WORK_CELLS: usize = 2_000_000;

#[derive(Clone)]
struct Group {
    heads: Vec<(usize, f64)>,
}

impl Group {
    fn valid(&self, source: &Source) -> bool {
        self.heads.len() == HEADS
            && self.heads.iter().enumerate().all(|(at, &(j, tau))| {
                j < source.spec.heads
                    && good(tau)
                    && tau > 0.0
                    && (at == 0 || self.heads[at - 1].0 < j)
            })
    }
}

#[derive(Default)]
pub(super) struct Report {
    pub(super) checked: usize,
    pub(super) added: usize,
    pub(super) complete_masks: usize,
    pub(super) work_cells: usize,
    pub(super) rank_trials: usize,
    prior_masks: usize,
    prior_work_cells: usize,
    pub(super) expired: bool,
    search_expired: bool,
    exhausted: bool,
}

impl Report {
    pub(super) fn receipt(&self) {
        eprintln!("[input-head-large-negative-group] checked={} added={} complete_masks={} total_complete_masks={} rank_trials={} work_cells={} total_work_cells={} search_expired={} phase_expired={} exhausted={}",
            self.checked,self.added,self.complete_masks,self.prior_masks+self.complete_masks,
            self.rank_trials,self.work_cells,self.prior_work_cells+self.work_cells,
            u8::from(self.search_expired),u8::from(self.expired),u8::from(self.exhausted));
    }
}

struct Budget {
    deadline: Instant,
    since_poll: usize,
    report: Report,
}

impl Budget {
    fn new(deadline: Instant, prior_masks: usize, prior_work_cells: usize) -> Self {
        Self {
            deadline,
            since_poll: 256,
            report: Report {
                prior_masks,
                prior_work_cells,
                ..Report::default()
            },
        }
    }

    fn charge(&mut self, cells: usize) -> bool {
        if cells == 0 || self.since_poll >= 256 {
            self.since_poll = 0;
            if !live(self.deadline) {
                self.report.expired = true;
                return false;
            }
        }
        let Some(next) = self.report.work_cells.checked_add(cells) else {
            self.report.exhausted = true;
            return false;
        };
        if self
            .report
            .prior_work_cells
            .checked_add(next)
            .is_none_or(|n| n > SHARED_WORK_CELLS)
        {
            self.report.exhausted = true;
            return false;
        }
        self.report.work_cells = next;
        self.since_poll = self.since_poll.saturating_add(cells);
        true
    }

    fn completed_mask(&mut self) -> bool {
        self.report.complete_masks += 1;
        #[cfg(test)]
        if EXPIRE_AFTER_MASK.with(|fault| {
            if fault.get() == Some(self.report.complete_masks) {
                fault.set(None);
                true
            } else {
                false
            }
        }) {
            self.deadline = Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap();
        }
        self.charge(0)
    }
}

fn source_valid(source: &Source) -> bool {
    let d = source.spec.inputs;
    let h = source.spec.heads;
    source.lo.len() == d
        && source.hi.len() == d
        && source
            .lo
            .iter()
            .zip(&source.hi)
            .all(|(&lo, &hi)| I { lo, hi }.valid())
        && d.checked_mul(h) == Some(source.spec.first.len())
        && source.spec.first_bias.len() == h
        && source.first_error.len() == h
        && source.first_error.iter().all(|&e| good(e) && e >= 0.0)
        && source.q.len() == h
        && source.y.len() == h
        && source.q.iter().all(|q| q.valid())
        && source.y.iter().all(|y| y.valid())
}

/// Full independent bias/error balls and EVERY original input coordinate.
/// Lower support of the exact negative represented seed, then negation,
/// encloses the upper support of this ONE mask. Tiny products are charged by I.
fn mask_upper(source: &Source, group: &Group, mask: usize, budget: &mut Budget) -> Option<f64> {
    // The immutable Source was fully shape/error qualified by the caller.
    if !group.valid(source) || mask >= (1usize << HEADS) || !budget.charge(0) || !fp_qualified() {
        return None;
    }
    let d = source.spec.inputs;
    let h = source.spec.heads;
    if !budget.charge(d) {
        return None;
    }
    let mut affine = Affine::zero(d);
    for (at, &(head, tau)) in group.heads.iter().enumerate() {
        if mask & (1 << at) == 0 {
            continue;
        }
        if !budget.charge(2) {
            return None;
        }
        let seed = I::p(-tau);
        let error = I::p(tau).mul(I::p(source.first_error[head]))?.hi;
        affine.b = affine
            .b
            .add(seed.mul(I::p(f64::from(source.spec.first_bias[head])))?)?
            .add(I {
                lo: -error,
                hi: error,
            })?;
        for k in 0..d {
            if !budget.charge(1) {
                return None;
            }
            affine.a[k] =
                affine.a[k].add(seed.mul(I::p(f64::from(source.spec.first[k * h + head])))?)?;
        }
    }
    if !budget.charge(d) {
        return None;
    }
    let boxes = source
        .lo
        .iter()
        .zip(&source.hi)
        .map(|(&lo, &hi)| I { lo, hi })
        .collect::<Vec<_>>();
    let upper = -affine.lower(&boxes, budget.deadline)?;
    (budget.charge(0) && fp_qualified() && good(upper)).then_some(upper)
}

fn owned_group_row(source: &Source, group: &Group, budget: &mut Budget) -> Option<OwnedRow> {
    let validation_cells = source
        .spec
        .heads
        .checked_mul(3)?
        .checked_add(source.spec.inputs)?;
    if !budget.charge(validation_cells)
        || !group.valid(source)
        || !source_valid(source)
        || !budget.charge(0)
        || !fp_qualified()
    {
        return None;
    }
    let count = 1usize << HEADS;
    let used = budget
        .report
        .prior_masks
        .checked_add(budget.report.complete_masks)?;
    if used.checked_add(count)? > SHARED_MASKS {
        budget.report.exhausted = true;
        return None;
    }
    let mut upper = 0.0f64;
    for mask in 0..count {
        upper = upper.max(mask_upper(source, group, mask, budget)?);
        if !budget.completed_mask() {
            return None;
        }
    }
    let row = OwnedRow {
        coefficients: group
            .heads
            .iter()
            .map(|&(j, tau)| Some((source.spec.inputs.checked_add(j)?, I::p(-tau))))
            .collect::<Option<Vec<_>>>()?,
        lower: I::p(-upper),
    };
    (budget.charge(0)
        && fp_qualified()
        && row.valid(source.spec.inputs.checked_add(source.spec.heads)?))
    .then_some(row)
}

/// Separate wider-Source entry. Reuses the complete owned 32-mask checker;
/// no growth/ranking or old lower endpoint enters this function. Counters are
/// shared by the caller across all nonzero-dual retained recipes, including
/// work completed by an ultimately refused/expired candidate.
pub(super) fn source_lower(
    source: &Source,
    heads: &[(usize, f64)],
    deadline: Instant,
    masks: &mut usize,
    cells: &mut usize,
) -> Option<f64> {
    if heads.len() != HEADS
        || *masks > SHARED_MASKS
        || *cells > SHARED_WORK_CELLS
        || !live(deadline)
        || !fp_qualified()
    {
        return None;
    }
    let group = Group {
        heads: heads.to_vec(),
    };
    let mut budget = Budget::new(deadline, *masks, *cells);
    let result = owned_group_row(source, &group, &mut budget).map(|row| row.lower.lo);
    *masks = budget
        .report
        .prior_masks
        .checked_add(budget.report.complete_masks)?;
    *cells = budget
        .report
        .prior_work_cells
        .checked_add(budget.report.work_cells)?;
    (budget.charge(0) && fp_qualified())
        .then_some(result)
        .flatten()
}

/// Nominal independent-vs-mask debt is bounded search advice ONLY.
fn rank(source: &Source, heads: &[(usize, f64)], budget: &mut Budget) -> Option<f64> {
    if !(4..=HEADS).contains(&heads.len())
        || budget.report.rank_trials >= MAX_RANK_TRIALS
        || !budget.charge(0)
    {
        return None;
    }
    budget.report.rank_trials += 1;
    let mut upper = 0.0f64;
    for mask in 1..(1usize << heads.len()) {
        let mut value = 0.0;
        for (at, &(j, tau)) in heads.iter().enumerate() {
            if mask & (1 << at) != 0 {
                if !budget.charge(1) {
                    return None;
                }
                value += tau * f64::from(source.spec.first_bias[j]);
            }
        }
        for k in 0..source.spec.inputs {
            let mut coefficient = 0.0;
            for (at, &(j, tau)) in heads.iter().enumerate() {
                if mask & (1 << at) != 0 {
                    if !budget.charge(1) {
                        return None;
                    }
                    coefficient += tau * f64::from(source.spec.first[k * source.spec.heads + j]);
                }
            }
            if !budget.charge(1) {
                return None;
            }
            value += (coefficient * source.lo[k]).max(coefficient * source.hi[k]);
        }
        if !good(value) {
            return None;
        }
        upper = upper.max(value);
    }
    if !budget.charge(heads.len()) {
        return None;
    }
    let independent = heads
        .iter()
        .map(|&(j, tau)| tau * source.y[j].hi.max(0.0))
        .sum::<f64>();
    let score = independent - upper;
    (budget.charge(0) && good(score)).then_some(score)
}

fn same_group(a: &Group, b: &Group) -> bool {
    a.heads.len() == b.heads.len()
        && a.heads
            .iter()
            .zip(&b.heads)
            .all(|(&(i, a), &(j, b))| i == j && a.to_bits() == b.to_bits())
}

fn proposals(source: &Source, old_rows: &[OwnedRow], budget: &mut Budget) -> Vec<Group> {
    let Some(validation_cells) = source
        .spec
        .heads
        .checked_mul(3)
        .and_then(|n| n.checked_add(source.spec.inputs))
    else {
        return vec![];
    };
    if !budget.charge(validation_cells) || !source_valid(source) || !budget.charge(0) {
        return vec![];
    }
    let d = source.spec.inputs;
    let mut bases = Vec::new();
    for row in old_rows {
        if !budget.charge(row.coefficients.len() + 1) {
            break;
        }
        if row.coefficients.len() != 4 {
            continue;
        }
        let heads = row
            .coefficients
            .iter()
            .map(|&(j, a)| {
                let head = j.checked_sub(d)?;
                (head < source.spec.heads && good(a.lo) && a.lo == a.hi && a.hi < 0.0)
                    .then_some((head, -a.lo))
            })
            .collect::<Option<Vec<_>>>();
        let Some(heads) = heads else {
            continue;
        };
        if !heads
            .iter()
            .enumerate()
            .all(|(at, &(j, _))| at == 0 || heads[at - 1].0 < j)
        {
            continue;
        }
        if let Some(score) = rank(source, &heads, budget) {
            if !bases
                .iter()
                .any(|(_, old): &(f64, Vec<(usize, f64)>)| *old == heads)
            {
                bases.push((score, heads));
            }
        }
    }
    bases.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1[0].0.cmp(&b.1[0].0)));
    bases.truncate(MAX_BASES);
    if bases.is_empty() {
        return vec![];
    }
    let mut pool = Vec::new();
    for j in 0..source.spec.heads {
        if !budget.charge(1) {
            break;
        }
        let (q, y) = (source.q[j], source.y[j]);
        if q.hi >= 0.0 || y.lo >= 0.0 || y.hi <= 0.0 {
            continue;
        }
        let tau = -q.hi;
        let debt = tau * (-y.lo * y.hi / (y.hi - y.lo));
        if good(tau) && tau > 0.0 && good(debt) && debt > 0.0 {
            pool.push((debt, j, tau));
        }
    }
    pool.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    pool.truncate(MAX_POOL);
    let mut grown = Vec::new();
    for (_, base) in bases {
        let mut best: Option<(f64, Group)> = None;
        for &(_, j, tau) in &pool {
            if base.iter().any(|&(old, _)| old == j) {
                continue;
            }
            let mut group = Group {
                heads: base.clone(),
            };
            group.heads.push((j, tau));
            group.heads.sort_by_key(|&(j, _)| j);
            if let Some(score) = rank(source, &group.heads, budget) {
                if group.valid(source)
                    && score > 0.0
                    && best.as_ref().is_none_or(|(old, _)| score > *old)
                {
                    best = Some((score, group));
                }
            }
        }
        if let Some((score, group)) = best {
            if !grown.iter().any(|(_, old)| same_group(old, &group)) {
                grown.push((score, group));
            }
        }
        if budget.report.expired || budget.report.exhausted {
            break;
        }
    }
    grown.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.heads[0].0.cmp(&b.1.heads[0].0))
    });
    grown
        .into_iter()
        .take(MAX_GROWN)
        .map(|(_, group)| group)
        .collect()
}

pub(super) fn rows(
    source: &Source,
    old_rows: &[OwnedRow],
    prior_masks: usize,
    prior_work_cells: usize,
    deadline: Instant,
) -> (Vec<(OwnedRow, usize)>, Report) {
    let now = Instant::now();
    let search_deadline = now
        .checked_add(deadline.saturating_duration_since(now) / 2)
        .unwrap_or(deadline)
        .min(deadline);
    let mut budget = Budget::new(search_deadline, prior_masks, prior_work_cells);
    if prior_masks > SHARED_MASKS || prior_work_cells > SHARED_WORK_CELLS {
        budget.report.exhausted = true;
        return (vec![], budget.report);
    }
    let proposed = proposals(source, old_rows, &mut budget);
    budget.report.search_expired = budget.report.expired;
    budget.report.expired = false;
    budget.deadline = deadline;
    budget.since_poll = 256;
    let mut result = Vec::new();
    for group in proposed {
        if !budget.charge(0) {
            break;
        }
        if let Some(row) = owned_group_row(source, &group, &mut budget) {
            budget.report.checked += 1;
            result.push((row, 1usize << HEADS));
        }
        if budget.report.expired || budget.report.exhausted {
            break;
        }
    }
    (result, budget.report)
}

#[cfg(test)]
thread_local! { pub(super) static EXPIRE_AFTER_MASK:std::cell::Cell<Option<usize>>=const { std::cell::Cell::new(None) }; }

#[cfg(test)]
#[path = "input_head_large_negative_groups_tests.rs"]
mod tests;
