// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded search only; none of these scores certify geometry.
use super::*;

/// Nominal cancellation is search only. Keep neutral growth prefixes: no
/// proper cycle subset separates its mean-input/head-one outer point, even
/// though such a subset can improve independent per-head upper support.
#[cfg(test)]
pub(super) fn proposals(source: &Source, budget: &mut Budget) -> Option<Vec<Direction>> {
    let debt: Vec<_> = source.q.iter().map(|q| 0.5 * q.lo + 0.5 * q.hi).collect();
    groups(source, &debt, false, budget)
}

fn groups(
    source: &Source,
    debt: &[f64],
    positive_links: bool,
    budget: &mut Budget,
) -> Option<Vec<Direction>> {
    let d = source.spec.inputs;
    let h = source.spec.heads;
    let mut pool = Vec::new();
    if debt.len() != h {
        return None;
    }
    for (head, (&q, &y)) in debt.iter().zip(&source.y).enumerate() {
        budget.charge(1)?;
        if good(q) && q < 0.0 && y.lo < 0.0 && y.hi > 0.0 {
            let score = -q * (-y.lo).min(y.hi);
            if good(score) && score > 0.0 {
                pool.push((head, score, -q));
            }
        }
    }
    pool.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    pool.truncate(MAX_POOL);
    let n = pool.len();
    if n < MIN_SUPPORT {
        return Some(vec![]);
    }
    let largest = pool.iter().map(|item| item.2).fold(0.0, f64::max);
    let tau = |at: usize| {
        let ratio = pool[at].2 / largest;
        if ratio >= 0.75 {
            1.0
        } else if ratio >= 0.375 {
            0.5
        } else {
            0.25
        }
    };
    let mut links = vec![vec![0.0; n]; n];
    for a in 0..n {
        for b in a + 1..n {
            let mut score = 0.0;
            for input in 0..d {
                budget.charge(1)?;
                let wa = f64::from(source.spec.first[input * h + pool[a].0]);
                let wb = f64::from(source.spec.first[input * h + pool[b].0]);
                if wa.signum() != wb.signum() || positive_links {
                    score += wa.abs().min(wb.abs()) * (source.hi[input] - source.lo[input]);
                }
            }
            if good(score) && score > 0.0 {
                links[a][b] = score;
                links[b][a] = score;
            }
        }
    }
    let mut result = Vec::new();
    let mut append = |indices: &[usize]| {
        if !(MIN_SUPPORT..=MAX_SUPPORT).contains(&indices.len()) || result.len() >= MAX_CANDIDATES {
            return;
        }
        let mut heads: Vec<_> = indices.iter().map(|&at| (pool[at].0, tau(at))).collect();
        heads.sort_by_key(|&(head, _)| head);
        if !result.iter().any(|old: &Direction| old.heads == heads) {
            result.push(Direction { heads });
        }
    };
    // Whole small components are candidates, independently of source order.
    let mut seen = vec![false; n];
    for seed in 0..n {
        if seen[seed] {
            continue;
        }
        let mut component = vec![seed];
        seen[seed] = true;
        let mut at = 0;
        while at < component.len() {
            budget.charge(n)?;
            for next in 0..n {
                if !seen[next] && links[component[at]][next] > 0.0 {
                    seen[next] = true;
                    component.push(next);
                }
            }
            at += 1;
        }
        append(&component);
    }
    // Dense components need several seeds; no positive-gain pruning of prefixes.
    for seed in 0..n.min(16) {
        let mut selected = vec![seed];
        while selected.len() < MAX_SUPPORT.min(n) {
            budget.charge(n.checked_mul(selected.len())?)?;
            let next = (0..n)
                .filter(|at| !selected.contains(at))
                .max_by(|&a, &b| {
                    let score = |at| selected.iter().map(|&old| links[old][at]).sum::<f64>();
                    score(a)
                        .total_cmp(&score(b))
                        .then_with(|| pool[b].0.cmp(&pool[a].0))
                })?;
            selected.push(next);
            if selected.len() == MAX_SUPPORT - 1 {
                append(&selected);
            }
        }
        append(&selected);
    }
    budget.charge(0)?;
    Some(result)
}

#[derive(Clone)]
pub(super) struct Candidate {
    pub(super) direction: Direction,
    pub(super) eta: Vec<(usize, f64)>,
}

impl Candidate {
    pub(super) fn key(&self) -> Vec<u64> {
        let mut key = Vec::new();
        for &(j, a) in &self.direction.heads {
            key.extend([j as u64, a.to_bits()]);
        }
        key.push(u64::MAX);
        for &(j, a) in &self.eta {
            key.extend([j as u64, a.to_bits()]);
        }
        key
    }
}

/// Canonical coordinate projections of sum(tau*W). These finite point
/// coefficients are choices of geometry, not rounded certificate endpoints.
fn eta_variants(
    source: &Source,
    direction: &Direction,
    gradient: &[f64],
    budget: &mut Budget,
) -> Option<Vec<(Vec<(usize, f64)>, f64)>> {
    if gradient.len() != source.spec.inputs {
        return None;
    }
    let mut ranked = Vec::new();
    for (j, &g) in gradient.iter().enumerate() {
        budget.charge(1)?;
        let mut value = 0.0;
        for &(head, tau) in &direction.heads {
            budget.charge(1)?;
            value += tau * f64::from(source.spec.first[j * source.spec.heads + head]);
        }
        let score = value * g * (source.hi[j] - source.lo[j]);
        if good(value) && value != 0.0 && good(score) {
            ranked.push((j, value, score));
        }
    }
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    Some(
        ranked
            .into_iter()
            .take(2)
            .map(|(j, a, score)| (vec![(j, a)], score))
            .collect(),
    )
}

pub(super) fn proposals_for_debt_excluding(
    source: &Source,
    debt: &[f64],
    gradient: &[f64],
    seen: &[Vec<u64>],
    budget: &mut Budget,
) -> Option<Vec<Candidate>> {
    let groups = groups(source, debt, true, budget)?;
    let mut candidates = Vec::new();
    for direction in groups {
        let etas = eta_variants(source, &direction, gradient, budget)?;
        // A microscopic interval-only eta=0 improvement must not consume the
        // scarce batch before a positively aligned affine variant of that SAME
        // support. Zero/cancelling eta still retains the historical first slot.
        let mut variants: Vec<_> = etas
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(eta, _)| eta.clone())
            .collect();
        variants.push(vec![]);
        variants.extend(
            etas.into_iter()
                .filter(|(_, score)| *score <= 0.0)
                .map(|(eta, _)| eta),
        );
        for eta in variants {
            let candidate = Candidate {
                direction: direction.clone(),
                eta,
            };
            // Filter BEFORE the per-round cap. Otherwise old tried variants
            // permanently hide later groups even after a new feedback round.
            if !seen.contains(&candidate.key()) {
                candidates.push(candidate);
            }
            if candidates.len() == MAX_CANDIDATES {
                return Some(candidates);
            }
        }
    }
    Some(candidates)
}

pub(super) fn initial_gradient(source: &Source, budget: &mut Budget) -> Option<Vec<f64>> {
    let slopes = source.slopes()?;
    let mut gradient = vec![0.0; source.spec.inputs];
    for (j, g) in gradient.iter_mut().enumerate() {
        for (head, &slope) in slopes.iter().enumerate() {
            budget.charge(1)?;
            *g += slope * f64::from(source.spec.first[j * source.spec.heads + head]);
        }
        if !good(*g) {
            return None;
        }
    }
    Some(gradient)
}

pub(super) fn track_candidates(
    source: &Source,
    base: &[f64],
    debt: &[f64],
    gradient: &[f64],
    seen: &[Vec<u64>],
    budget: &mut Budget,
) -> Option<Vec<Candidate>> {
    // The Source-CROWN anchor preserves meaningful coordinate alignment even
    // when a valid but degenerate LP dual overcancels common head coefficients.
    let gradient: Vec<_> = base
        .iter()
        .zip(gradient)
        .map(|(&base, &now)| {
            if base == 0.0 {
                now
            } else {
                base + 0.5 * now.clamp(-base.abs(), base.abs())
            }
        })
        .collect();
    budget.charge(gradient.len())?;
    proposals_for_debt_excluding(source, debt, &gradient, seen, budget)
}
