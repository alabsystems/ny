// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Breadth-first current-G advice; one complete original-q replay owns acceptance.
use super::*;

pub(super) struct Accepted {
    pub refined: Vec<bool>,
    pub pending_zero: Vec<bool>,
}
#[derive(Clone, Copy)]
struct Visit {
    index: usize,
    minimum: usize,
}
pub(super) fn overhead(p: &HullMarginProblem, plan: &Plan) -> Option<usize> {
    if !capacity(p) || plan.components.is_empty() || plan.components.len() > p.hidden_dim {
        return None;
    }
    let alpha = plan.components.iter().try_fold(0usize, |v, c| {
        if c.negative.is_empty() || c.negative.len() > 8 || c.positive.len() > MAX_POSITIVE {
            return None;
        }
        v.checked_add(c.alpha.len())
    })?;
    // Source/plan scans, quote ordering, retained bits and all bounded descriptors.
    p.input_dim
        .checked_mul(p.hidden_dim)?
        .checked_add(alpha)?
        .checked_add(p.hidden_dim)?
        .checked_add(plan.components.len().checked_mul(plan.components.len())?)?
        .checked_add(1)?
        .checked_mul(32)
}
fn pressured(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    cost: usize,
) -> Option<Vec<Visit>> {
    if b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || b.search_end > b.authority
        || !live(b.search_end)
        || !b.can_check()
    {
        return None;
    }
    b.charge(overhead(cuts.problem, plan)?)?;
    shape(cuts)?;
    plan.valid(cuts)?;
    if plan.cost(cuts.problem)? != cost {
        return None;
    }
    if b.dimensions
        != (
            cuts.problem.input_dim,
            cuts.problem.hidden_dim,
            cuts.problem.c.iter().filter(|c| **c > 0.).count(),
        )
    {
        return None;
    }
    let mut visits = Vec::with_capacity(plan.components.len());
    let mut legacy_work = 0usize;
    let mut legacy_checks = 0usize;
    for (index, c) in plan.components.iter().enumerate() {
        if c.positive.is_empty() {
            continue;
        }
        let work = search::work(cuts.problem, c)?;
        let seeds = if c.g.iter().any(|v| *v != 0.) {
            2usize
        } else {
            1usize
        };
        legacy_work = legacy_work.checked_add(
            work.cost(work.positives)?
                .checked_add(cost)?
                .checked_mul(seeds)?,
        )?;
        legacy_checks = legacy_checks.checked_add(seeds)?;
        visits.push(Visit {
            index,
            minimum: work.cost(1)?,
        });
    }
    if visits.len() < 2 {
        return None;
    }
    let slots = MAX_CHECKS.checked_sub(b.state.lock().ok()?.checks)?;
    // Preserve the existing schedule when every complete current/zero seed fits.
    if legacy_work <= b.remaining()? && legacy_checks <= slots {
        return None;
    }
    Some(visits)
}
fn select(mut visits: Vec<Visit>, available: usize) -> Option<Vec<Visit>> {
    // If all minima cannot fit, the cheapest complete-mask visits form a prefix.
    // Equal quotes retain original component order; no fixture/index recognizer.
    visits.sort_by_key(|v| (v.minimum, v.index));
    let mut used = 0usize;
    let mut count = 0usize;
    for v in &visits {
        let next = used.checked_add(v.minimum)?;
        if next > available {
            break;
        }
        used = next;
        count += 1;
    }
    visits.truncate(count);
    (!visits.is_empty()).then_some(visits)
}
fn stage(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    visits: &[Visit],
    b: &Budget,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Plan, Vec<bool>)> {
    let mut future = visits
        .iter()
        .try_fold(0usize, |v, c| v.checked_add(c.minimum))?;
    let mut trial = plan.clone();
    let mut visited = vec![false; plan.components.len()];
    for (slot, v) in visits.iter().enumerate() {
        if !live(b.search_end) || !poll() {
            return None;
        }
        future = future.checked_sub(v.minimum)?;
        let room = b.remaining()?.checked_sub(future)?;
        let extra = room.checked_sub(v.minimum)? / (visits.len() - slot);
        let allowance = v.minimum.checked_add(extra)?;
        let component =
            search::optimize_prepaid(cuts, &trial.components[v.index], b, allowance, poll)?;
        // Only a fully completed component can enter this still-untrusted batch.
        trial.components[v.index] = component;
        visited[v.index] = true;
    }
    Some((trial, visited))
}
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    cuts: &HullRootMipCuts<'_>,
    plan: &mut Plan,
    current: &mut f64,
    best: &mut Option<SignedMajorant>,
    floor: f64,
    cost: usize,
    b: &Budget,
    seal: &Arc<RetainedWords>,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Accepted> {
    if *current > 0. || !poll() {
        return None;
    }
    let visits = pressured(cuts, plan, b, cost)?;
    // Prepay the entire eventual checker AND its search slot before any advice.
    // Publication reserves remain untouched. A refused batch spends an attempt.
    b.proof(cost, false)?;
    let visits = select(visits, b.remaining()?)?;
    let (trial, visited) = stage(cuts, plan, &visits, b, poll)?;
    let checked = certificate::check(cuts, &trial, b, b.search_end, poll)?;
    if checked.root_lower <= *current {
        return None;
    }
    let refined: Vec<_> = trial
        .components
        .iter()
        .zip(&plan.components)
        .map(|(new, old)| {
            new.alpha
                .iter()
                .map(|v| v.to_bits())
                .ne(old.alpha.iter().map(|v| v.to_bits()))
        })
        .collect();
    let pending_zero = visited
        .iter()
        .zip(&trial.components)
        .map(|(&done, c)| done && c.g.iter().any(|v| *v != 0.))
        .collect();
    let lower = checked.root_lower;
    let mut counts = b.state.lock().ok()?;
    counts.refined_components = counts
        .refined_components
        .checked_add(refined.iter().filter(|v| **v).count())?;
    drop(counts);
    *plan = trial;
    *current = lower;
    // Drop the replaced plan before constructing the new retained witness;
    // the peak large-vector copy count stays within the existing payload quote.
    if lower > floor {
        *best = Some(row(cuts, plan, checked, cost, b, seal));
    }
    Some(Accepted {
        refined,
        pending_zero,
    })
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_batch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_batch_retention_tests.rs"]
mod retention_tests;
