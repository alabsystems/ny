// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete bounded all-mask support, anchored to the original attained g.
//! This is real-arithmetic HullMarginProblem replay, not OriginalReplay/E1.
use super::*;
use std::sync::Arc;
#[path = "single_hidden_hull_mip_group_budget.rs"]
mod budget;
pub(super) use budget::*;
#[path = "single_hidden_hull_mip_group_search.rs"]
mod search;
pub(super) use search::refine;

#[derive(Debug, Clone)]
pub(super) struct GroupMajorant(Arc<Data>);
#[derive(Debug)]
struct Data {
    source: usize,
    bits: Vec<u64>,
    heads: Vec<usize>,
    weights: Vec<f64>,
    g: Vec<f64>,
    upper: f64,
    _memory: Permit,
}
impl GroupMajorant {
    pub(super) fn heads(&self) -> &[usize] {
        &self.0.heads
    }
    pub(super) fn g(&self) -> &[f64] {
        &self.0.g
    }
    pub(super) fn upper(&self) -> f64 {
        self.0.upper
    }
    pub(super) fn mask_count(&self) -> usize {
        1 << self.0.heads.len()
    }
    pub(super) fn owned_by(&self, budget: &Budget) -> bool {
        Arc::ptr_eq(&self.0._memory.budget.state, &budget.state)
            && self.0._memory.budget.authority == budget.authority
            && self.0._memory.budget.search_end == budget.search_end
    }
    pub(super) fn matches(&self, cuts: &HullRootMipCuts<'_>) -> Option<()> {
        let _scratch = self
            .0
            ._memory
            .budget
            .memory((2 * self.0.heads.len() + cuts.problem.input_dim) * 8 + 128)?;
        let (heads, weights, g) = proposal(cuts, &self.0.heads)?;
        (self.0.source == std::ptr::from_ref(cuts.problem) as usize
            && source_matches(cuts.problem, &heads, &self.0.bits)
            && heads == self.0.heads
            && same(&weights, &self.0.weights)
            && same(&g, &self.0.g))
        .then_some(())
    }
}
fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn source_bits(p: &HullMarginProblem, heads: &[usize]) -> Vec<u64> {
    p.lower
        .iter()
        .chain(&p.upper)
        .copied()
        .chain(
            heads
                .iter()
                .flat_map(|&j| p.w_row(j).iter().copied().chain([p.b1[j]])),
        )
        .map(f64::to_bits)
        .collect()
}
fn source_matches(p: &HullMarginProblem, heads: &[usize], bits: &[u64]) -> bool {
    let mut values = p.lower.iter().chain(&p.upper).copied().chain(
        heads
            .iter()
            .flat_map(|&j| p.w_row(j).iter().copied().chain([p.b1[j]])),
    );
    bits.iter()
        .all(|&b| values.next().is_some_and(|v| v.to_bits() == b))
        && values.next().is_none()
}
fn proposal(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize],
) -> Option<(Vec<usize>, Vec<f64>, Vec<f64>)> {
    let p = cuts.problem;
    if !capacity(p)
        || heads.len() < 2
        || heads.len() > MAX_GROUP
        || p.w1.len() != p.input_dim.checked_mul(p.hidden_dim)?
        || p.b1.len() != p.hidden_dim
        || p.c.len() != p.hidden_dim
        || p.lower.len() != p.input_dim
        || p.upper.len() != p.input_dim
        || p.lower
            .iter()
            .zip(&p.upper)
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return None;
    }
    let mut heads = heads.to_vec();
    heads.sort_unstable();
    if heads.windows(2).any(|v| v[0] == v[1]) {
        return None;
    }
    let mut g = vec![0.; p.input_dim];
    let mut weights = Vec::new();
    for &j in &heads {
        let a = finite(-*p.c.get(j)?)?;
        if a <= 0. {
            return None;
        }
        let plane = cuts.planes.iter().find(|r| r.head == j)?;
        let source = cuts
            .multipliers
            .g
            .get(j * p.input_dim..(j + 1) * p.input_dim)?;
        if plane.g.len() != g.len() || !same(&plane.g, source) {
            return None;
        }
        for (v, &x) in g.iter_mut().zip(source) {
            *v = finite(*v + a * x)?;
        }
        weights.push(a);
    }
    Some((heads, weights, g))
}

pub(super) fn support(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize],
    budget: &Budget,
    poll: &mut dyn FnMut() -> bool,
) -> Option<GroupMajorant> {
    budget.attempt()?;
    kernel(cuts, heads, budget, poll, false)
}
fn kernel(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize],
    budget: &Budget,
    poll: &mut dyn FnMut() -> bool,
    replay: bool,
) -> Option<GroupMajorant> {
    require_hull_environment().ok()?;
    if !poll() {
        return None;
    }
    let p = cuts.problem;
    let d = p.input_dim;
    let k = heads.len();
    if !(2..=MAX_GROUP).contains(&k) || !capacity(p) {
        return None;
    }
    let _scratch = budget.memory((2 * k + d).checked_mul(8)?.checked_add(128)?)?;
    // Charge validation, point summation, products and retained copies BEFORE work.
    budget.charge(
        (k + 1)
            .checked_mul(d + 1)?
            .checked_mul(6)?
            .checked_add(p.hidden_dim * k)?,
        0,
        replay,
    )?;
    let (heads, weights, g) = proposal(cuts, heads)?;
    let n = 1usize.checked_shl(k as u32)?;
    let stride = d + 1;
    let _temp = budget.memory(
        n.checked_mul(stride)?
            .checked_mul(16)?
            .checked_add(k.checked_mul(stride)?.checked_mul(16)?)?,
    )?;
    let mut products = vec![[0.; 2]; k * stride];
    for (bit, &j) in heads.iter().enumerate() {
        if !live(cuts.deadline) || !poll() {
            return None;
        }
        for i in 0..stride {
            let v = if i == d { p.b1[j] } else { p.w_row(j)[i] };
            products[bit * stride + i] = [
                finite(mul_down(weights[bit], v))?,
                finite(mul_up(weights[bit], v))?,
            ];
        }
    }
    let mut table = vec![[0.; 2]; n * stride];
    for i in 0..d {
        table[i] = [-g[i], -g[i]];
    }
    let mut best = f64::NEG_INFINITY;
    for mask in 0..n {
        budget.charge(stride.checked_mul(8)?, 1, replay)?;
        if !live(cuts.deadline) || !poll() {
            return None;
        }
        if mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let parent = mask & (mask - 1);
            let (before, now) = table.split_at_mut(mask * stride);
            for i in 0..stride {
                let x = before[parent * stride + i];
                let a = products[bit * stride + i];
                now[i] = [finite(add_down(x[0], a[0]))?, finite(add_up(x[1], a[1]))?];
            }
        }
        let row = &table[mask * stride..(mask + 1) * stride];
        let mut upper = row[d][1];
        for i in 0..d {
            let [lo, hi] = row[i];
            let l = p.lower[i];
            let u = p.upper[i];
            let v = mul_up(lo, l)
                .max(mul_up(lo, u))
                .max(mul_up(hi, l))
                .max(mul_up(hi, u));
            upper = finite(add_up(upper, finite(v)?))?;
        }
        best = best.max(upper);
    }
    if !live(if replay {
        budget.authority
    } else {
        budget.search_end
    }) || !poll()
    {
        return None;
    }
    require_hull_environment().ok()?;
    let memory = budget.memory(
        (2 * k + 3 * d + k * stride)
            .checked_mul(8)?
            .checked_add(256)?,
    )?;
    let bits = source_bits(p, &heads);
    Some(GroupMajorant(Arc::new(Data {
        source: std::ptr::from_ref(p) as usize,
        bits,
        heads,
        weights,
        g,
        upper: best,
        _memory: memory,
    })))
}
pub(super) fn replay(
    cuts: &HullRootMipCuts<'_>,
    row: &GroupMajorant,
    budget: &Budget,
    poll: &mut dyn FnMut() -> bool,
) -> Option<GroupMajorant> {
    if !row.owned_by(budget) {
        return None;
    }
    budget.charge(
        (row.heads().len() + 1)
            .checked_mul(cuts.problem.input_dim + 1)?
            .checked_mul(3)?
            .checked_add(cuts.problem.hidden_dim * row.heads().len())?,
        0,
        true,
    )?;
    row.matches(cuts)?;
    let new = kernel(cuts, row.heads(), budget, poll, true)?;
    (same(new.g(), row.g()) && new.upper().to_bits() == row.upper().to_bits()).then_some(new)
}
pub(super) fn replay_cost(
    cuts: &HullRootMipCuts<'_>,
    rows: &[GroupMajorant],
) -> Option<(usize, usize)> {
    let p = cuts.problem;
    let mut work = 16usize.checked_mul((p.hidden_dim + 1).checked_mul(p.input_dim + 1)?)?;
    let mut masks = 0usize;
    for row in rows {
        let k = row.heads().len();
        let n = row.mask_count();
        masks = masks.checked_add(n)?;
        work = work
            .checked_add((k + 1).checked_mul(p.input_dim + 1)?.checked_mul(12)?)?
            .checked_add(3 * p.hidden_dim * k)?
            .checked_add(n.checked_mul(p.input_dim + 1)?.checked_mul(8)?)?;
    }
    Some((work, masks))
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_group_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_group_tests.rs"]
mod tests;
