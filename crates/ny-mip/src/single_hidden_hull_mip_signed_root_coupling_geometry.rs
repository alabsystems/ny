// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact original mass reassignment; nominal point directions are not proof.
use super::*;

pub(super) struct Built {
    pub plan: Plan,
    pub selected: usize,
    pub recipient: Option<usize>,
}
fn energy(p: &HullMarginProblem, j: usize) -> Option<f64> {
    let mut v = p.b1.get(j)?.abs();
    for i in 0..p.input_dim {
        v = finite(v + p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
    }
    finite(v * p.c.get(j)?.abs())
}
pub(super) fn build(
    cuts: &HullRootMipCuts<'_>,
    old: &Plan,
    choice: &resources::Choice,
    b: &Budget,
    t: &Ticket,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Built> {
    t.claim_build(cuts, b, old, choice)?;
    let p = cuts.problem;
    let d = p.input_dim;
    let (parents, parent_count) = t.parents(cuts, b, choice)?;
    let parents = &parents[..parent_count];
    let mut negative = Vec::with_capacity(16);
    let mut appearances = 0;
    let mut whole = true;
    for &index in parents {
        let part = old.components.get(index)?;
        appearances += part.positive.len();
        whole &= part.positive_units.iter().all(|u| *u == 2);
        if negative.len().checked_add(part.negative.len())? > 16 {
            return None;
        }
        negative.extend_from_slice(&part.negative);
    }
    if appearances > 96 || !whole && (parent_count != 2 || appearances > 32) {
        return None;
    }
    if negative.len() > 16
        || choice.selected == 0
        || choice.selected > 6
        || choice.selected > negative.len()
    {
        return None;
    }
    let mut ranked = Vec::with_capacity(negative.len());
    for &j in &negative {
        t.alive(cuts, b, poll)?;
        ranked.push((j, energy(p, j)?));
    }
    let mut selected = Vec::with_capacity(choice.selected);
    for _ in 0..choice.selected {
        let mut best: Option<(usize, f64)> = None;
        for &(j, e) in &ranked {
            if !selected.contains(&j) && best.is_none_or(|(_, old)| e > old) {
                best = Some((j, e));
            }
        }
        selected.push(best?.0);
    }
    let tail: Vec<_> = negative
        .iter()
        .copied()
        .filter(|j| !selected.contains(j))
        .collect();
    let mut positive = Vec::with_capacity(appearances);
    let mut units: Vec<u8> = Vec::with_capacity(positive.capacity());
    for &index in parents {
        let part = &old.components[index];
        for (&j, &u) in part.positive.iter().zip(&part.positive_units) {
            t.alive(cuts, b, poll)?;
            // Whole-unit parents are disjoint under the sealed Plan.valid
            // certificate, so this path has no quadratic duplicate scan.
            if whole {
                positive.push(j);
                units.push(u);
            } else if let Some(slot) = positive.iter().position(|k| *k == j) {
                units[slot] = units[slot].checked_add(u)?;
            } else {
                positive.push(j);
                units.push(u);
            }
        }
    }
    if positive.is_empty() || positive.len() > 96 || units.iter().any(|u| !matches!(u, 1 | 2)) {
        return None;
    }
    // ONE cached original-affine energy scan and bounded stable selection sort.
    // Split/active ordering is advice only; all actual mass stays in the Plan.
    let mut ranked_positive = Vec::with_capacity(positive.len());
    for (slot, &j) in positive.iter().enumerate() {
        t.alive(cuts, b, poll)?;
        ranked_positive.push((
            slot,
            finite(energy(p, j)? * (f64::from(units[slot]) * 0.5))?,
        ));
    }
    for i in 0..ranked_positive.len().min(16) {
        let mut best = i;
        for j in i + 1..ranked_positive.len() {
            if ranked_positive[j].1 > ranked_positive[best].1 {
                best = j;
            }
        }
        // Rotating preserves the original order of all equal-energy entries.
        ranked_positive[i..=best].rotate_right(1);
    }
    let ordered_positive: Vec<_> = ranked_positive.iter().map(|(i, _)| positive[*i]).collect();
    let ordered_units: Vec<_> = ranked_positive.iter().map(|(i, _)| units[*i]).collect();
    positive = ordered_positive;
    units = ordered_units;
    let take = positive.len().min(16);
    if !whole && positive.len() > take && tail.is_empty() {
        return None;
    }
    let mut parts = Vec::with_capacity(old.components.len() - parent_count + 1 + tail.len());
    let selected_index = parents[0];
    let pa = positive[..take].to_vec();
    let pu = units[..take].to_vec();
    let seed: Vec<_> = pa
        .iter()
        .map(|&j| sanitize_alpha(cuts.multipliers.alpha[j]))
        .collect();
    let mut merged = Some(Component {
        negative: selected,
        positive: pa,
        positive_units: pu,
        g: vec![0.; d],
        alpha: seed.repeat(1usize << choice.selected),
    });
    // Stable replacement at the earliest removed parent: unrelated component
    // order is preserved, including deterministic later mass-allocation ties.
    for (index, part) in old.components.iter().enumerate() {
        t.alive(cuts, b, poll)?;
        if index == parents[0] {
            parts.push(merged.take()?);
        }
        if !parents.contains(&index) {
            parts.push(part.clone());
        }
    }
    let mut recipient = None;
    for (slot, &j) in tail.iter().enumerate() {
        t.alive(cuts, b, poll)?;
        let mut part = Component {
            negative: vec![j],
            positive: Vec::new(),
            positive_units: Vec::new(),
            g: Vec::with_capacity(d),
            alpha: Vec::new(),
        };
        for i in 0..d {
            part.g.push(finite(p.c[j] * cuts.multipliers.g[j * d + i])?);
        }
        if !whole && slot == 0 && positive.len() > take {
            part.positive = positive[take..].to_vec();
            part.positive_units = units[take..].to_vec();
            let seed: Vec<_> = part
                .positive
                .iter()
                .map(|&j| sanitize_alpha(cuts.multipliers.alpha[j]))
                .collect();
            part.alpha = seed.repeat(2);
            for (pos, &head) in part.positive.iter().enumerate() {
                let weight = part.nominal_coefficient(p, pos)?;
                for i in 0..d {
                    part.g[i] = finite(part.g[i] + weight * seed[pos] * p.w_row(head)[i])?;
                }
            }
            recipient = Some(parts.len());
        }
        parts.push(part);
    }
    let extra = if whole { positive.len() - take } else { 0 };
    let mut leftover = Vec::with_capacity(old.leftover.len().checked_add(extra)?);
    leftover.extend_from_slice(&old.leftover);
    if whole {
        leftover.extend_from_slice(&positive[take..]);
    }
    let mut plan = Plan {
        components: parts,
        leftover,
    };
    // All remainder directions, including leftover positive affine terms, are
    // proposals. The whole interval checker prices the COMPLETE sum residual.
    let mut remainder = vec![0.; d];
    for (index, part) in plan.components.iter().enumerate() {
        if index == selected_index {
            continue;
        }
        t.alive(cuts, b, poll)?;
        for i in 0..d {
            remainder[i] = finite(remainder[i] + part.g[i])?;
        }
    }
    for &j in &plan.leftover {
        t.alive(cuts, b, poll)?;
        let weight = finite(p.c[j] * sanitize_alpha(cuts.multipliers.alpha[j]))?;
        for i in 0..d {
            remainder[i] = finite(remainder[i] + weight * p.w_row(j)[i])?;
        }
    }
    for g in &mut remainder {
        *g = -*g;
    }
    plan.components[selected_index].g = remainder;
    plan.valid(cuts)?;
    let new_words = resources::words(old)?
        .checked_add(4096)?
        .checked_add(14usize.checked_mul(d.checked_add(5)?)?)?;
    if payload::heap(&plan)? > payload::plan_bound(new_words, old.components.len())? {
        return None;
    }
    if plan.cost(p)? > choice.check
        || plan.components[selected_index].positive.len() > choice.positive
        || recipient.is_some_and(|i| plan.components[i].positive.len() > choice.recipient)
    {
        return None;
    }
    Some(Built {
        plan,
        selected: selected_index,
        recipient,
    })
}
