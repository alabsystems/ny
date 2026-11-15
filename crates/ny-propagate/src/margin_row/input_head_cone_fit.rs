// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded affine cone search, with NO certificate authority. All numbers
//! here choose represented masses only. The parent recomputes the full owned
//! signed source contraction and copies the SAME masses into the ReLU row.
//! A rounded small least-squares residual is never a source identity.

use super::*;

const MAX_TARGETS: usize = 16;
const MAX_COLUMNS: usize = 64;
const MAX_SWEEPS: usize = 8;
const MAX_WORK_CELLS: usize = 2_000_000;
const MAX_LOCAL_TIME: Duration = Duration::from_millis(20);

#[derive(Default)]
pub(super) struct Report {
    targets: usize,
    sweeps: usize,
    coordinates: usize,
    work_cells: usize,
    support_max: usize,
    pub(super) checked: usize,
    pub(super) added: usize,
    pub(super) expired: bool,
    work_exhausted: bool,
}

impl Report {
    pub(super) fn receipt(&self) {
        eprintln!("[input-head-cone-fit] targets={} sweeps={} coordinates={} work_cells={} support_max={} checked={} added={} phase_expired={} work_exhausted={}",
            self.targets,self.sweeps,self.coordinates,self.work_cells,self.support_max,
            self.checked,self.added,u8::from(self.expired),u8::from(self.work_exhausted));
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

    fn charge(&mut self, count: usize) -> bool {
        // Poll each bounded tile rather than reading the clock at every
        // multiply-add. Boundary calls with count0 always poll immediately.
        if count == 0 || self.since_poll >= 256 {
            self.since_poll = 0;
            if !live(self.deadline) {
                self.report.expired = true;
                return false;
            }
        }
        let Some(next) = self.report.work_cells.checked_add(count) else {
            self.report.work_exhausted = true;
            return false;
        };
        if next > self.cap {
            self.report.work_exhausted = true;
            return false;
        }
        self.report.work_cells = next;
        self.since_poll = self.since_poll.saturating_add(count);
        true
    }
}

pub(super) fn phase_deadline(now: Instant, original: Instant) -> Instant {
    #[cfg(test)]
    if FORCE_EXPIRED_PHASE.with(|fault| fault.replace(false)) {
        return now.checked_sub(Duration::from_millis(1)).unwrap_or(now);
    }
    let remaining = original.saturating_duration_since(now);
    now.checked_add((remaining / 3).min(MAX_LOCAL_TIME))
        .unwrap_or(now)
        .min(original)
}

/// Leave one quarter of the local phase for complete owned row pricing.
/// Neither subphase extends the original invocation/caller deadline.
pub(super) fn search_deadline(now: Instant, phase_end: Instant) -> Instant {
    let remaining = phase_end.saturating_duration_since(now);
    now.checked_add(remaining.checked_sub(remaining / 4).unwrap())
        .unwrap_or(now)
        .min(phase_end)
}

struct Column {
    head: usize,
    values: Vec<f64>,
    norm: f64,
    capacity: f64,
}

fn vector(source: &Source, head: usize, budget: &mut Budget) -> Option<Vec<f64>> {
    if head >= source.spec.heads || !budget.charge(0) {
        return None;
    }
    let mut result = Vec::with_capacity(source.spec.inputs + 1);
    let mut centered_bias = f64::from(source.spec.first_bias[head]);
    for k in 0..source.spec.inputs {
        if !budget.charge(2) {
            return None;
        }
        // Weighted full affine directions, including the bias. This scaling
        // is search advice only; it is NEVER used by the source certificate.
        let center = 0.5 * source.lo[k] + 0.5 * source.hi[k];
        let radius = 0.5 * source.hi[k] - 0.5 * source.lo[k];
        let weight = f64::from(source.spec.first[k * source.spec.heads + head]);
        let value = weight * radius;
        centered_bias += weight * center;
        if !good(center) || !good(radius) || radius < 0.0 || !good(value) || !good(centered_bias) {
            return None;
        }
        result.push(value);
    }
    if !budget.charge(1) {
        return None;
    }
    result.push(centered_bias);
    Some(result)
}

fn dot(a: &[f64], b: &[f64], budget: &mut Budget) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let mut result = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        if !budget.charge(1) {
            return None;
        }
        result += a * b;
    }
    good(result).then_some(result)
}

fn add_scaled(dst: &mut [f64], a: f64, src: &[f64], budget: &mut Budget) -> Option<()> {
    if dst.len() != src.len() || !good(a) {
        return None;
    }
    for (dst, &src) in dst.iter_mut().zip(src) {
        if !budget.charge(1) {
            return None;
        }
        *dst += a * src;
        if !good(*dst) {
            return None;
        }
    }
    Some(())
}

fn same_cut(a: &MassCut, b: &MassCut) -> bool {
    a.negative == b.negative
        && a.tau.to_bits() == b.tau.to_bits()
        && a.positive.len() == b.positive.len()
        && a.positive.iter().all(|&(j, value)| {
            b.positive
                .iter()
                .any(|&(k, other)| j == k && value.to_bits() == other.to_bits())
        })
}

fn score(source: &Source, cut: &MassCut, budget: &mut Budget) -> Option<f64> {
    // Reserve all scalar source-ranking cells before its bounded full-width
    // loop. This counter does not measure final certified source replay.
    let cells = (source.spec.inputs + 1).checked_mul(cut.positive.len() + 1)?;
    if !budget.charge(cells) {
        return None;
    }
    let deadline = budget.deadline;
    let mut terms = cut.positive.clone();
    terms.push((cut.negative, -cut.tau));
    let lower = nominal_difference(source, &terms, deadline)?;
    let independent = cut
        .positive
        .iter()
        .map(|&(j, a)| a * source.y[j].lo.max(0.0))
        .sum::<f64>()
        - cut.tau * source.y[cut.negative].hi.max(0.0);
    let result = lower - independent;
    good(result).then_some(result)
}

/// Coordinate descent on an ordinary bounded nonnegative least-squares
/// heuristic. Source dimensions/capacities only guide it; no convergence,
/// KKT condition, exact span, or optimality claim enters the checked cut.
fn fit(
    source: &Source,
    negative: usize,
    legacy: &[MassCut],
    budget: &mut Budget,
) -> Option<MassCut> {
    if !budget.charge(0) {
        return None;
    }
    let tau = -source.q[negative].hi;
    if !good(tau) || tau <= 0.0 {
        return None;
    }
    let direction = vector(source, negative, budget)?;
    let target: Vec<_> = direction.iter().map(|a| tau * a).collect();
    if target.iter().any(|&x| !good(x)) {
        return None;
    }
    let warm = legacy.iter().find(|cut| cut.negative == negative);
    let mut columns = Vec::new();
    for positive in 0..source.spec.heads {
        if source.q[positive].lo <= 0.0 || positive == negative {
            continue;
        }
        let values = vector(source, positive, budget)?;
        let norm = dot(&values, &values, budget)?;
        if norm <= 0.0 {
            continue;
        }
        let correlation = dot(&values, &target, budget)? / norm.sqrt();
        if !good(correlation) {
            continue;
        }
        let old = warm.is_some_and(|cut| cut.positive.iter().any(|&(j, _)| j == positive));
        columns.push((
            old,
            correlation,
            Column {
                head: positive,
                values,
                norm,
                capacity: source.q[positive].lo,
            },
        ));
    }
    // Retain the old warm support before filling the remaining search pool.
    columns.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(b.1.total_cmp(&a.1))
            .then(a.2.head.cmp(&b.2.head))
    });
    columns.truncate(MAX_COLUMNS);
    if columns.is_empty() {
        return None;
    }
    let columns: Vec<_> = columns.into_iter().map(|(_, _, column)| column).collect();
    let mut masses = vec![0.0; columns.len()];
    let mut residual: Vec<_> = target.iter().map(|a| -a).collect();
    let mut support = 0usize;
    if let Some(warm) = warm {
        for (at, column) in columns.iter().enumerate() {
            if let Some(&(_, a)) = warm.positive.iter().find(|&&(j, _)| j == column.head) {
                let point = a.clamp(0.0, column.capacity);
                if point > 0.0 && support < MAX_POSITIVE_MASSES {
                    masses[at] = point;
                    support += 1;
                    add_scaled(&mut residual, point, &column.values, budget)?;
                }
            }
        }
    }
    for _ in 0..MAX_SWEEPS {
        if !budget.charge(0) {
            return None;
        }
        budget.report.sweeps += 1;
        let mut changed = false;
        for (at, column) in columns.iter().enumerate() {
            let gradient = dot(&column.values, &residual, budget)?;
            let point = (masses[at] - gradient / column.norm).clamp(0.0, column.capacity);
            if !good(point) {
                return None;
            }
            budget.report.coordinates += 1;
            if point == masses[at]
                || (masses[at] == 0.0 && point > 0.0 && support >= MAX_POSITIVE_MASSES)
            {
                continue;
            }
            add_scaled(&mut residual, point - masses[at], &column.values, budget)?;
            if masses[at] == 0.0 && point > 0.0 {
                support += 1;
            }
            if masses[at] > 0.0 && point == 0.0 {
                support -= 1;
            }
            masses[at] = point;
            changed = true;
        }
        if !changed || dot(&residual, &residual, budget)? == 0.0 {
            break;
        }
    }
    let mut positive: Vec<_> = columns
        .iter()
        .zip(masses)
        .filter_map(|(column, a)| (a > 0.0).then_some((column.head, a)))
        .collect();
    positive.sort_by_key(|&(j, _)| j);
    if positive.is_empty() || positive.len() > MAX_POSITIVE_MASSES || !budget.charge(0) {
        return None;
    }
    budget.report.support_max = budget.report.support_max.max(positive.len());
    Some(MassCut {
        positive,
        negative,
        tau,
    })
}

pub(super) fn proposals(
    source: &Source,
    legacy: &[MassCut],
    deadline: Instant,
) -> (Vec<MassCut>, Report) {
    let mut budget = Budget::new(deadline, MAX_WORK_CELLS);
    let result = propose_with_budget(source, legacy, &mut budget);
    (result, budget.report)
}

fn propose_with_budget(source: &Source, legacy: &[MassCut], budget: &mut Budget) -> Vec<MassCut> {
    let mut negatives: Vec<_> = (0..source.spec.heads)
        .filter(|&j| source.q[j].hi < 0.0)
        .collect();
    negatives.sort_by(|&a, &b| source.q[a].hi.total_cmp(&source.q[b].hi).then(a.cmp(&b)));
    negatives.truncate(MAX_TARGETS);
    let mut result = Vec::new();
    for negative in negatives {
        if !budget.charge(0) {
            break;
        }
        budget.report.targets += 1;
        let Some(cut) = fit(source, negative, legacy, budget) else {
            if budget.report.expired || budget.report.work_exhausted {
                break;
            }
            continue;
        };
        if legacy.iter().chain(&result).any(|old| same_cut(old, &cut)) {
            continue;
        }
        let Some(value) = score(source, &cut, budget) else {
            continue;
        };
        let old_score = legacy
            .iter()
            .filter(|old| old.negative == negative)
            .filter_map(|old| score(source, old, budget))
            .fold(0.0, f64::max);
        if value > old_score && good(value) && budget.charge(0) {
            result.push(cut);
        }
    }
    result
}

#[cfg(test)]
thread_local! {
    static FORCE_EXPIRED_PHASE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[path = "input_head_cone_fit_tests.rs"]
mod tests;
