// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private joint upper rows on genuine authored ReLU outputs.
//! For represented tau>=0, sum(tau*rho(y))=max_mask sum(tau*mask*y).
//! Every mask receives a complete full-source affine support BEFORE a row
//! exists. Search ranks groups only; source-owned interval replay is authority.

use super::*;

const MAX_GROUPS: usize = 16;
const MAX_HEADS: usize = 4;
const MAX_POOL: usize = 16;
const MAX_RANK_TRIALS: usize = 512;
const MAX_WORK_CELLS: usize = 2_000_000;
const MAX_COMPLETE_MASKS: usize = MAX_GROUPS * (1 << MAX_HEADS);

#[derive(Clone)]
struct Group {
    heads: Vec<(usize, f64)>,
}

impl Group {
    fn valid(&self, source: &Source) -> bool {
        (2..=MAX_HEADS).contains(&self.heads.len())
            && self.heads.iter().enumerate().all(|(at, &(j, tau))| {
                j < source.spec.heads
                    && good(tau)
                    && tau >= 0.0
                    && (at == 0 || self.heads[at - 1].0 < j)
            })
    }
}

#[derive(Default)]
pub(super) struct Report {
    pub(super) checked: usize,
    pub(super) added: usize,
    complete_masks: usize,
    rank_trials: usize,
    work_cells: usize,
    pub(super) expired: bool,
    search_expired: bool,
    work_exhausted: bool,
}

impl Report {
    pub(super) fn completed_masks(&self) -> usize {
        self.complete_masks
    }
    pub(super) fn work_cells(&self) -> usize {
        self.work_cells
    }

    pub(super) fn receipt(&self) {
        eprintln!("[input-head-negative-group] checked={} added={} complete_masks={} rank_trials={} work_cells={} search_expired={} phase_expired={} work_exhausted={}",
            self.checked,self.added,self.complete_masks,self.rank_trials,self.work_cells,
            u8::from(self.search_expired),u8::from(self.expired),u8::from(self.work_exhausted));
    }
}

struct Budget {
    deadline: Instant,
    cap: usize,
    since_poll: usize,
    report: Report,
}

impl Budget {
    fn new(deadline: Instant, cap: usize) -> Self {
        Self {
            deadline,
            cap,
            since_poll: 256,
            report: Report::default(),
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
            self.report.work_exhausted = true;
            return false;
        };
        if next > self.cap {
            self.report.work_exhausted = true;
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
            // Named deterministic seam after REAL completed mask support.
            // Expiry applies to this local attempt, never a partial row.
            self.deadline = Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap();
        }
        self.charge(0)
    }
}

/// Complete directed upper support for ONE affine mask. No imported row,
/// phase assumption, nominal cancellation or parent Y endpoint substitutes
/// for this full W/b/E1 contraction. Negative seed lower then exact negation
/// is the same outward upper theorem without duplicating interval kernels.
fn mask_upper(source: &Source, group: &Group, mask: usize, budget: &mut Budget) -> Option<f64> {
    let d = source.spec.inputs;
    let h = source.spec.heads;
    if !group.valid(source)
        || mask >= 1usize.checked_shl(group.heads.len() as u32)?
        || source.spec.first.len() != d.checked_mul(h)?
        || source.spec.first_bias.len() != h
        || source.first_error.len() != h
        || source.lo.len() != d
        || source.hi.len() != d
        || !budget.charge(0)
        || !fp_qualified()
    {
        return None;
    }
    if !budget.charge(d) {
        return None;
    }
    let mut affine = Affine::zero(d);
    for (at, &(head, tau)) in group.heads.iter().enumerate() {
        if mask & (1 << at) == 0 || tau == 0.0 {
            continue;
        }
        if !budget.charge(2) || !good(source.first_error[head]) || source.first_error[head] < 0.0 {
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
    let lower = affine.lower(&boxes, budget.deadline)?;
    let upper = -lower;
    (budget.charge(0) && fp_qualified() && good(upper)).then_some(upper)
}

fn owned_group_row(source: &Source, group: &Group, budget: &mut Budget) -> Option<OwnedRow> {
    if !group.valid(source) || !budget.charge(0) || !fp_qualified() {
        return None;
    }
    let count = 1usize.checked_shl(group.heads.len() as u32)?;
    if budget.report.complete_masks.checked_add(count)? > MAX_COMPLETE_MASKS {
        return None;
    }
    let mut upper = 0.0f64;
    for mask in 0..count {
        let value = mask_upper(source, group, mask, budget)?;
        upper = upper.max(value);
        if !budget.completed_mask() {
            return None;
        }
    }
    // Only ALL masks authorize this row. A completed prefix MAX is never
    // published, cached as a row, or priced by the final weak-dual checker.
    let row = OwnedRow {
        coefficients: group
            .heads
            .iter()
            .filter(|&&(_, tau)| tau != 0.0)
            .map(|&(j, tau)| (source.spec.inputs + j, I::p(-tau)))
            .collect(),
        lower: I::p(-upper),
    };
    (budget.charge(0)
        && fp_qualified()
        && row.valid(source.spec.inputs.checked_add(source.spec.heads)?))
    .then_some(row)
}

fn same_group(a: &Group, b: &Group) -> bool {
    a.heads.len() == b.heads.len()
        && a.heads
            .iter()
            .zip(&b.heads)
            .all(|(&(i, a), &(j, b))| i == j && a.to_bits() == b.to_bits())
}

/// Ordinary nominal mask geometry for ranking ONLY. Missing phase/rounding
/// precision here cannot weaken an owned row because every accepted group
/// is independently re-contracted with interval arithmetic above.
fn rank(source: &Source, group: &Group, budget: &mut Budget) -> Option<f64> {
    if !group.valid(source) || budget.report.rank_trials >= MAX_RANK_TRIALS || !budget.charge(0) {
        return None;
    }
    budget.report.rank_trials += 1;
    let mut joint = 0.0f64;
    let h = source.spec.heads;
    for mask in 1..(1usize << group.heads.len()) {
        let mut value = 0.0;
        for (at, &(j, tau)) in group.heads.iter().enumerate() {
            if mask & (1 << at) != 0 {
                if !budget.charge(1) {
                    return None;
                }
                value += tau * f64::from(source.spec.first_bias[j]);
            }
        }
        for k in 0..source.spec.inputs {
            let mut coefficient = 0.0;
            for (at, &(j, tau)) in group.heads.iter().enumerate() {
                if mask & (1 << at) != 0 {
                    if !budget.charge(1) {
                        return None;
                    }
                    coefficient += tau * f64::from(source.spec.first[k * h + j]);
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
        joint = joint.max(value);
    }
    if !budget.charge(group.heads.len()) {
        return None;
    }
    let independent = group
        .heads
        .iter()
        .map(|&(j, tau)| tau * source.y[j].hi.max(0.0))
        .sum::<f64>();
    let score = independent - joint;
    (budget.charge(0) && good(score)).then_some(score)
}

fn proposals(source: &Source, budget: &mut Budget) -> Vec<Group> {
    if source.q.len() != source.spec.heads
        || source.y.len() != source.spec.heads
        || source.q.iter().any(|q| !q.valid())
        || source.y.iter().any(|y| !y.valid())
        || source.lo.len() != source.spec.inputs
        || source.hi.len() != source.spec.inputs
        || source.spec.first.len() != source.spec.inputs.saturating_mul(source.spec.heads)
        || source.spec.first_bias.len() != source.spec.heads
    {
        return vec![];
    }
    let mut pool = Vec::new();
    for j in 0..source.spec.heads {
        if !budget.charge(1) {
            break;
        }
        let q = source.q[j];
        let y = source.y[j];
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
    if pool.len() < 2 {
        return vec![];
    }
    let mut pairs = Vec::new();
    for a in 0..pool.len() {
        for b in a + 1..pool.len() {
            let mut heads = vec![(pool[a].1, pool[a].2), (pool[b].1, pool[b].2)];
            heads.sort_by_key(|&(j, _)| j);
            let group = Group { heads };
            if let Some(score) = rank(source, &group, budget) {
                if score > 0.0 {
                    pairs.push((score, group));
                }
            }
            if budget.report.expired || budget.report.work_exhausted {
                break;
            }
        }
        if budget.report.expired || budget.report.work_exhausted {
            break;
        }
    }
    pairs.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.heads[0].0.cmp(&b.1.heads[0].0))
            .then(a.1.heads[1].0.cmp(&b.1.heads[1].0))
    });
    let mut result = pairs
        .iter()
        .take(8)
        .map(|(_, g)| g.clone())
        .collect::<Vec<_>>();
    for (_, pair) in pairs.iter().take(4) {
        let mut group = pair.clone();
        for size in [3, 4] {
            let mut best: Option<(f64, Group)> = None;
            for &(_, j, tau) in &pool {
                if group.heads.iter().any(|&(old, _)| old == j) {
                    continue;
                }
                let mut next = group.clone();
                next.heads.push((j, tau));
                next.heads.sort_by_key(|&(j, _)| j);
                if next.heads.len() != size {
                    continue;
                }
                if let Some(score) = rank(source, &next, budget) {
                    if score > 0.0 && best.as_ref().is_none_or(|(old, _)| score > *old) {
                        best = Some((score, next));
                    }
                }
            }
            let Some((_, next)) = best else {
                break;
            };
            group = next;
            if !result.iter().any(|old| same_group(old, &group)) && result.len() < MAX_GROUPS {
                result.push(group.clone());
            }
            if budget.report.expired || budget.report.work_exhausted {
                break;
            }
        }
        if budget.report.expired || budget.report.work_exhausted {
            break;
        }
    }
    result
}

pub(super) fn eligible(source: &Source) -> bool {
    source.q.len() == source.spec.heads
        && source.y.len() == source.spec.heads
        && source
            .q
            .iter()
            .zip(&source.y)
            .filter(|&(q, y)| q.valid() && y.valid() && q.hi < 0.0 && y.lo < 0.0 && y.hi > 0.0)
            .take(2)
            .count()
            == 2
}

pub(super) fn rows(source: &Source, deadline: Instant) -> (Vec<(OwnedRow, usize)>, Report) {
    if !eligible(source) {
        return (vec![], Report::default());
    }
    let now = Instant::now();
    let search_deadline = now
        .checked_add(deadline.saturating_duration_since(now) / 2)
        .unwrap_or(deadline)
        .min(deadline);
    let mut budget = Budget::new(search_deadline, MAX_WORK_CELLS);
    let proposed = proposals(source, &mut budget);
    // An expired SEARCH can still yield a completed shortlist. Mask replay
    // uses only the reserved remainder of the original joint-stage window.
    budget.report.search_expired = budget.report.expired;
    budget.report.expired = false;
    budget.deadline = deadline;
    budget.since_poll = 256;
    let mut result = Vec::new();
    for group in proposed.into_iter().take(MAX_GROUPS) {
        if !budget.charge(0) {
            break;
        }
        if let Some(row) = owned_group_row(source, &group, &mut budget) {
            budget.report.checked += 1;
            result.push((row, 1usize << group.heads.len()));
        }
        if budget.report.expired || budget.report.work_exhausted {
            break;
        }
    }
    (result, budget.report)
}

#[cfg(test)]
thread_local! { static EXPIRE_AFTER_MASK:std::cell::Cell<Option<usize>>=const { std::cell::Cell::new(None) }; }

#[cfg(test)]
#[path = "input_head_negative_groups_tests.rs"]
mod tests;
