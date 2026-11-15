// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Geometry-based allocation proposals. Only independent signed proof grants authority.
use super::*;

#[path = "input_head_directional_allocation.rs"]
mod allocation;

#[derive(Clone)]
pub(super) struct Candidate {
    pub(super) negative: Direction,
    pub(super) positive: signed::PositiveTerms,
}

pub(super) struct Column {
    head: usize,
    mass: f64,
    values: Vec<f64>,
    norm: f64,
}

struct Search<'a> {
    budget: &'a mut Budget,
    start: usize,
    maximum: usize,
}

impl Search<'_> {
    fn room(&self, cells: usize) -> bool {
        self.budget
            .cells
            .checked_sub(self.start)
            .and_then(|n| n.checked_add(cells))
            .is_some_and(|n| n <= self.maximum)
    }
    fn charge(&mut self, cells: usize) -> Option<()> {
        if self
            .budget
            .cells
            .checked_sub(self.start)?
            .checked_add(cells)?
            > self.maximum
        {
            return None;
        }
        self.budget.charge(cells)
    }
}

fn column(source: &Source, head: usize, mass: f64, work: &mut Search<'_>) -> Option<Column> {
    let d = source.spec.inputs;
    work.charge(d.checked_mul(3)?.checked_add(1)?)?;
    let mut values = Vec::with_capacity(d);
    let mut norm = 0.0;
    for j in 0..d {
        if j & 31 == 0 {
            work.charge(0)?;
        }
        let radius = 0.5 * source.hi[j] - 0.5 * source.lo[j];
        let value = radius * f64::from(source.spec.first[j * source.spec.heads + head]);
        if !good(radius) || radius < 0.0 || !good(value) {
            return None;
        }
        norm += value * value;
        values.push(value);
    }
    good(norm).then_some(Column {
        head,
        mass,
        values,
        norm,
    })
}

fn distance(a: &Column, b: &Column, work: &mut Search<'_>) -> Option<f64> {
    work.charge(a.values.len().checked_add(4)?)?;
    let dot: f64 = a.values.iter().zip(&b.values).map(|(a, b)| a * b).sum();
    let denominator = a.norm.sqrt() * b.norm.sqrt();
    let value = if denominator > 0.0 {
        1.0 - dot / denominator
    } else {
        1.0
    };
    good(value).then_some(value.clamp(0.0, 2.0))
}

fn clusters(columns: &[Column], work: &mut Search<'_>) -> Option<Vec<Vec<usize>>> {
    let k = columns.len().checked_add(MAX_SUPPORT - 1)? / MAX_SUPPORT;
    if !(2..=MAX_CUTS * MAX_ROUNDS).contains(&k) {
        return None;
    }
    let mut seeds = vec![0];
    let mut nearest = vec![f64::INFINITY; columns.len()];
    while seeds.len() < k {
        let last = *seeds.last()?;
        let mut best = None;
        for at in 0..columns.len() {
            nearest[at] = nearest[at].min(distance(&columns[at], &columns[last], work)?);
            if !seeds.contains(&at) && best.is_none_or(|old| nearest[at] > nearest[old]) {
                best = Some(at);
            }
        }
        seeds.push(best?);
    }
    let mut groups: Vec<Vec<usize>> = (0..k).map(|_| Vec::with_capacity(MAX_SUPPORT)).collect();
    for at in 0..columns.len() {
        let mut best: Option<(usize, f64)> = None;
        for (g, &seed) in seeds.iter().enumerate() {
            if groups[g].len() == MAX_SUPPORT {
                continue;
            }
            let value = distance(&columns[at], &columns[seed], work)?;
            if best.is_none_or(|(_, old)| value < old) {
                best = Some((g, value));
            }
        }
        groups[best?.0].push(at);
    }
    Some(groups)
}

/// A discovery invariant, not a premise of any signed geometry certificate.
/// Independent manually overlapping rows remain valid and require full-q replay.
pub(super) fn valid_allocation(
    source: &Source,
    candidates: &[Candidate],
    budget: &mut Budget,
) -> bool {
    let mut check = || -> Option<()> {
        if source.q.len() != source.spec.heads || source.q.iter().any(|q| !q.valid()) {
            return None;
        }
        if candidates.is_empty() || candidates.len() > MAX_CUTS * MAX_ROUNDS {
            return None;
        }
        budget.charge(source.spec.heads.checked_mul(2)?)?;
        let mut seen = vec![false; source.spec.heads];
        let mut totals = vec![I::p(0.0); source.spec.heads];
        for candidate in candidates {
            if candidate.negative.heads.is_empty()
                || candidate.negative.heads.len() > MAX_SUPPORT
                || candidate.positive.0.is_empty()
            {
                return None;
            }
            for (at, &(head, tau)) in candidate.negative.heads.iter().enumerate() {
                budget.charge(2)?;
                if head >= seen.len()
                    || seen[head]
                    || !good(tau)
                    || tau <= 0.0
                    || source.q[head].hi >= 0.0
                    || (at > 0 && candidate.negative.heads[at - 1].0 >= head)
                {
                    return None;
                }
                seen[head] = true;
            }
            for (at, &(head, a)) in candidate.positive.0.iter().enumerate() {
                budget.charge(2)?;
                if head >= totals.len()
                    || !good(a)
                    || a <= 0.0
                    || source.q[head].lo <= 0.0
                    || (at > 0 && candidate.positive.0[at - 1].0 >= head)
                {
                    return None;
                }
                totals[head] = totals[head].add(I::p(a))?;
            }
        }
        for (head, total) in totals.iter().enumerate() {
            budget.charge(1)?;
            if total.hi > source.q[head].lo.max(0.0) {
                return None;
            }
        }
        Some(())
    };
    check().is_some()
}

/// All retained groups and omitted objective terms remain in the same full-q LP.
pub(super) fn discover(source: &Source, budget: &mut Budget) -> Vec<Candidate> {
    discover_checked(source, budget).unwrap_or_default()
}

fn discover_checked(source: &Source, budget: &mut Budget) -> Option<Vec<Candidate>> {
    let (d, h) = (source.spec.inputs, source.spec.heads);
    budget.charge(h.checked_mul(3)?)?;
    if source.q.len() != h
        || source.y.len() != h
        || source.lo.len() != d
        || source.hi.len() != d
        || source.spec.first.len() != d.checked_mul(h)?
    {
        return None;
    }
    let mut negatives = Vec::new();
    let mut positives = Vec::new();
    for head in 0..h {
        let q = source.q[head];
        if q.lo > 0.0 && source.y[head].hi > 0.0 {
            positives.push((head, q.lo));
        }
        if q.hi < 0.0 && source.y[head].lo < 0.0 && source.y[head].hi > 0.0 {
            let tau = -signed::point(q)?;
            let score = tau * source.y[head].hi;
            if good(tau) && tau > 0.0 && good(score) {
                negatives.push((head, tau, score));
            }
        }
    }
    // Preserve the original four-theta path for one group, including old controls.
    if negatives.len() <= MAX_SUPPORT || positives.is_empty() {
        return None;
    }
    let (negative_columns, groups, positive_columns) = {
        let start = budget.cells;
        let mut work = Search {
            budget,
            start,
            maximum: 125_000,
        };
        let log = usize::BITS as usize - h.max(1).leading_zeros() as usize;
        work.charge(h.checked_mul(log.checked_add(2)?)?)?;
        negatives.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        negatives.truncate(MAX_SUPPORT * MAX_CUTS * MAX_ROUNDS);
        let negative_columns = negatives
            .iter()
            .map(|&(head, mass, _)| column(source, head, mass, &mut work))
            .collect::<Option<Vec<_>>>()?;
        let groups = clusters(&negative_columns, &mut work)?;
        let positive_columns = positives
            .iter()
            .map(|&(head, mass)| column(source, head, mass, &mut work))
            .collect::<Option<Vec<_>>>()?;
        (negative_columns, groups, positive_columns)
    };
    let masses;
    let mut result;
    {
        let start = budget.cells;
        let mut work = Search {
            budget,
            start,
            maximum: 125_000,
        };
        masses = allocation::fit(&negative_columns, &groups, &positive_columns, &mut work)?;
        result = Vec::with_capacity(groups.len());
        for (g, indices) in groups.iter().enumerate() {
            let mut heads: Vec<_> = indices
                .iter()
                .map(|&at| (negative_columns[at].head, negative_columns[at].mass))
                .collect();
            work.charge(heads.len().checked_mul(4)?)?;
            heads.sort_by_key(|&(head, _)| head);
            work.charge(positive_columns.len())?;
            let positive = positive_columns
                .iter()
                .enumerate()
                .filter_map(|(p, col)| (masses[p][g] > 0.0).then_some((col.head, masses[p][g])))
                .collect::<Vec<_>>();
            if !positive.is_empty() {
                result.push(Candidate {
                    negative: Direction { heads },
                    positive: signed::PositiveTerms(positive),
                });
            }
        }
    }
    (result.len() >= 2 && valid_allocation(source, &result, budget)).then_some(result)
}
