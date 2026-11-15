// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Optional exact half-mass seed; all scores and staged masks are only advice.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SignedMassReport {
    pub attempted: bool,
    pub positive_appearances: usize,
    pub split_positive_heads: usize,
    pub baseline_checks: usize,
    pub final_checks: usize,
    pub funded_visits: usize,
    pub completed_visits: usize,
    pub selected: bool,
    pub banked_lower: Option<f64>,
    pub lower: Option<f64>,
    pub stop_reason: &'static str,
}
impl Default for SignedMassReport {
    fn default() -> Self {
        Self {
            attempted: false,
            positive_appearances: 0,
            split_positive_heads: 0,
            baseline_checks: 0,
            final_checks: 0,
            funded_visits: 0,
            completed_visits: 0,
            selected: false,
            banked_lower: None,
            lower: None,
            stop_reason: "not-attempted",
        }
    }
}
fn note(b: &Budget, apply: impl FnOnce(&mut SignedMassReport)) {
    if let Ok(mut c) = b.state.lock() {
        apply(&mut c.mass);
    }
}
fn stop(b: &Budget, reason: &'static str) {
    note(b, |m| m.stop_reason = reason);
}

pub(super) fn pressured(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    cost: usize,
    b: &Budget,
) -> Option<bool> {
    if plan.components.len() < 2 || b.dimensions.2 == 0 {
        return Some(false);
    }
    b.charge(
        cuts.problem
            .hidden_dim
            .checked_add(plan.components.len())?
            .checked_add(cuts.problem.input_dim)?
            .checked_add(1)?
            .checked_mul(16)?,
    )?;
    let mut work = direction::quote(cuts.problem, plan)?
        .work
        .checked_add(cost)?
        .checked_add(batch::overhead(cuts.problem, plan)?)?;
    let mut checks = 1usize;
    for c in &plan.components {
        if c.positive.is_empty() {
            continue;
        }
        let q = search::work(cuts.problem, c)?;
        // Direction advice may make an originally zero G nonzero. Quote BOTH
        // bounded seeds even then; a failed quote never proves affordability.
        let seeds = 2usize;
        work = work.checked_add(q.cost(q.positives)?.checked_add(cost)?.checked_mul(seeds)?)?;
        checks = checks.checked_add(seeds)?;
    }
    let slots = MAX_CHECKS.checked_sub(b.state.lock().ok()?.checks)?;
    Some(work > b.remaining()? || checks > slots)
}

/// Top-two affinity over an already selected negative partition. No fixture or
/// source-index recognition, and no claim that the partition discovery is best.
pub(super) fn plan(cuts: &HullRootMipCuts<'_>, whole: &Plan, b: &Budget) -> Option<Plan> {
    let p = cuts.problem;
    let (d, h, positives) = b.dimensions;
    if b.source != std::ptr::from_ref(p) as usize
        || b.authority > cuts.deadline
        || (d, h, positives)
            != (
                p.input_dim,
                p.hidden_dim,
                p.c.iter().filter(|c| **c > 0.).count(),
            )
    {
        return None;
    }
    let negatives = whole
        .components
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(c.negative.len()))?;
    let cost = positives
        .checked_mul(negatives)?
        .checked_mul(d + 1)?
        .checked_add(d.checked_mul(h)?.checked_add(h)?.checked_add(1)?)?
        .checked_mul(8)?
        .checked_add(
            positives
                .checked_mul(whole.components.len())?
                .checked_add(h + d + 1)?
                .checked_mul(16)?,
        )?;
    b.admit_sharing()?;
    b.charge(cost)?;
    shape(cuts)?;
    whole.valid(cuts)?;
    if whole
        .components
        .iter()
        .any(|c| c.positive_units.iter().any(|u| *u != 2))
    {
        return None;
    }
    let mut components = Vec::with_capacity(whole.components.len());
    for old in &whole.components {
        let mut c = Component {
            negative: old.negative.clone(),
            positive: Vec::new(),
            positive_units: Vec::new(),
            g: vec![0.; d],
            alpha: Vec::new(),
        };
        // Original per-head directions seed the proposal, never the certificate.
        for &j in &c.negative {
            for i in 0..d {
                c.g[i] = finite(c.g[i] + p.c[j] * cuts.multipliers.g[j * d + i])?;
            }
        }
        components.push(c);
    }
    let mut leftover = Vec::new();
    let mut split = false;
    for j in 0..h {
        if !live(b.search_end) {
            return None;
        }
        if p.c[j] <= 0. {
            continue;
        }
        let mut choices: [Option<(usize, f64)>; 2] = [None, None];
        for (index, c) in components.iter().enumerate() {
            if c.positive.len() >= MAX_POSITIVE {
                continue;
            }
            let score = c
                .negative
                .iter()
                .fold(0f64, |s, &k| s.max(geometry::affinity(p, j, k)));
            if score <= 0. {
                continue;
            }
            if choices[0].is_none_or(|(_, old)| score > old) {
                choices[1] = choices[0];
                choices[0] = Some((index, score));
            } else if choices[1].is_none_or(|(_, old)| score > old) {
                choices[1] = Some((index, score));
            }
        }
        match (choices[0], choices[1]) {
            (Some((a, _)), Some((b, _))) => {
                for index in [a, b] {
                    components[index].positive.push(j);
                    components[index].positive_units.push(1);
                }
                split = true;
            }
            (Some((index, _)), None) => {
                components[index].positive.push(j);
                components[index].positive_units.push(2);
            }
            _ => leftover.push(j),
        }
    }
    if !split {
        return None;
    }
    for c in &mut components {
        let initial: Vec<_> = c
            .positive
            .iter()
            .map(|&j| sanitize_alpha(cuts.multipliers.alpha[j]))
            .collect();
        c.alpha = initial.repeat(1usize << c.negative.len());
        for (slot, (&j, &alpha)) in c.positive.iter().zip(&initial).enumerate() {
            let k = c.nominal_coefficient(p, slot)?;
            for i in 0..d {
                c.g[i] = finite(c.g[i] + k * alpha * p.w_row(j)[i])?;
            }
        }
    }
    let plan = Plan {
        components,
        leftover,
    };
    plan.valid(cuts)?;
    note(b, |m| {
        m.positive_appearances = plan.components.iter().map(|c| c.positive.len()).sum();
        m.split_positive_heads = m.positive_appearances - (positives - plan.leftover.len());
        m.stop_reason = "constructed";
    });
    Some(plan)
}

/// Bank a complete baseline, then fund one complete-mask coordinate per chosen
/// component and ONE whole-plan check. A weaker baseline can still be refined.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    cuts: &HullRootMipCuts<'_>,
    selected: &mut Plan,
    current: &mut f64,
    best: &mut Option<SignedMajorant>,
    floor: f64,
    b: &Budget,
    seal: &Arc<RetainedWords>,
    poll: &mut dyn FnMut() -> bool,
) {
    if *current > 0.
        || selected.components.len() < 2
        || b.dimensions.2 == 0
        || !live(b.search_end)
        || !poll()
    {
        return;
    }
    note(b, |m| m.banked_lower = Some(*current));
    let Some(mut shared) = plan(cuts, selected, b) else {
        if b.state
            .lock()
            .is_ok_and(|c| c.mass.stop_reason == "proposal")
        {
            stop(b, "geometry-refused");
        }
        return;
    };
    let Some(cost) = shared.cost(cuts.problem) else {
        return;
    };
    let mut visits = Vec::new();
    for (index, c) in shared.components.iter().enumerate() {
        if c.positive.is_empty() {
            continue;
        }
        let Some(minimum) = search::work(cuts.problem, c).and_then(|q| q.cost(1)) else {
            return;
        };
        visits.push((minimum, index));
    }
    visits.sort_by_key(|v| *v);
    let overhead = cuts
        .problem
        .hidden_dim
        .checked_add(shared.components.len())
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(32));
    let Some(overhead) = overhead else {
        return;
    };
    if b.charge(overhead).is_none() {
        return;
    }
    let Some(slots) = b
        .state
        .lock()
        .ok()
        .and_then(|c| MAX_CHECKS.checked_sub(c.checks))
    else {
        return;
    };
    if slots < 2 {
        stop(b, "checks");
        return;
    }
    if b.reserve_at_least(cost).is_none() {
        stop(b, "reserve");
        return;
    }
    // Optional rows are prepaid before advice receives its share of work.
    let mut portfolio = visits
        .first()
        .and_then(|&(minimum, _)| portfolio::admit(cuts, &shared, b, seal, minimum));
    let Some(mut room) = cost
        .checked_mul(2)
        .and_then(|c| b.remaining()?.checked_sub(c))
    else {
        stop(b, "work");
        return;
    };
    let mut count = 0usize;
    for &(minimum, _) in &visits {
        if minimum > room {
            break;
        }
        room -= minimum;
        count += 1;
    }
    visits.truncate(count);
    if visits.is_empty() {
        stop(b, "minimum-refused");
        return;
    }
    note(b, |m| {
        m.funded_visits = visits.len();
        m.stop_reason = "funded";
    });
    if b.proof(cost, false).is_none() {
        stop(b, "baseline-credit");
        return;
    }
    let Some(initial) = certificate::check(cuts, &shared, b, b.search_end, poll) else {
        stop(b, "baseline-refused");
        return;
    };
    let local = initial.root_lower;
    note(b, |m| {
        m.baseline_checks += 1;
        m.lower = Some(local);
        m.stop_reason = "baseline-complete";
    });
    if local > *current {
        *current = local;
        *selected = shared.clone();
        if let Ok(mut c) = b.state.lock() {
            c.refined_components = 0;
            c.mass.selected = true;
        }
        if local > floor {
            *best = Some(row(cuts, &shared, initial, cost, b, seal));
        }
    }
    // This charge reserves the check AND its slot before any staged advice.
    if portfolio::pay_final(cuts, b, cost, portfolio.as_mut()).is_none() {
        stop(b, "final-credit");
        return;
    }
    let mut changed = 0usize;
    let Some(mut future) = visits.iter().try_fold(0usize, |n, v| n.checked_add(v.0)) else {
        stop(b, "quote");
        return;
    };
    for (slot, &(minimum, index)) in visits.iter().enumerate() {
        if !live(b.search_end) || !poll() {
            stop(b, "advice-refused");
            return;
        }
        let Some(next) = future.checked_sub(minimum) else {
            stop(b, "quote");
            return;
        };
        future = next;
        // The checker is already paid. Protect every FUTURE complete-mask
        // minimum, then share genuine spare credit; unused allowances carry on.
        let Some(room) = b.remaining().and_then(|v| v.checked_sub(future)) else {
            stop(b, "work");
            return;
        };
        let Some(extra) = room.checked_sub(minimum).map(|v| v / (visits.len() - slot)) else {
            stop(b, "work");
            return;
        };
        let Some(allowance) = minimum.checked_add(extra) else {
            stop(b, "quote");
            return;
        };
        let Some(component) =
            search::optimize_prepaid(cuts, &shared.components[index], b, allowance, poll)
        else {
            stop(b, "advice-refused");
            return;
        };
        if component
            .alpha
            .iter()
            .map(|a| a.to_bits())
            .ne(shared.components[index].alpha.iter().map(|a| a.to_bits()))
        {
            changed += 1;
        }
        shared.components[index] = component;
        note(b, |m| m.completed_visits += 1);
    }
    let Some(checked) = portfolio::check_final(cuts, &shared, b, portfolio, *current, poll) else {
        stop(b, "final-refused");
        return;
    };
    note(b, |m| {
        m.final_checks += 1;
        m.lower = Some(local.max(checked.root_lower));
        m.stop_reason = "checked-no-global-gain";
    });
    if checked.root_lower <= local || checked.root_lower <= *current {
        return;
    }
    *current = checked.root_lower;
    *selected = shared;
    if let Ok(mut c) = b.state.lock() {
        c.refined_components = changed;
        c.mass.selected = true;
        c.mass.stop_reason = "selected";
    }
    if *current > floor {
        *best = Some(row(cuts, selected, checked, cost, b, seal));
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_mass_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_mass_tests.rs"]
mod tests;
