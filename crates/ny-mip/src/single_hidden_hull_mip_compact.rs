// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One objective-weighted row from independently replayed disjoint pairs.
//! This deliberately trades the flexibility of many pair rows for one row.
use super::*;
use crate::ir::Col as IrCol;

#[derive(Debug)]
pub(super) struct Aggregate {
    pub(super) pairs: Vec<joint::JointMajorant>,
    pub(super) groups: Vec<group::GroupMajorant>,
    pub(super) group_budget: Option<group::Budget>,
    _group_memory: Option<group::Permit>,
    pub(super) g: Vec<f64>,
    pub(super) upper: f64,
    pub(super) improvement: f64,
}

/// Two bounded streaming matchings see the SAME completed pair proposals.
/// Replacement decisions are search only; neither matching certifies a row.
pub(super) struct Partitions {
    candidates: [Vec<joint::JointMajorant>; 2],
}

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}
fn same_slice(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(*x, *y))
}
fn overlaps(a: &[usize; 2], b: &[usize; 2]) -> bool {
    a.iter().any(|i| b.contains(i))
}

impl Partitions {
    pub(super) fn new() -> Self {
        Self {
            candidates: [Vec::new(), Vec::new()],
        }
    }
    pub(super) fn observe(&mut self, row: &joint::JointMajorant) {
        if !row.improvement.is_finite() || row.improvement <= 0.0 {
            return;
        }
        for (slot, rows) in self.candidates.iter_mut().enumerate() {
            let loss: f64 = rows
                .iter()
                .filter(|r| overlaps(&r.heads, &row.heads))
                .map(|r| r.improvement)
                .sum();
            // Different replacement thresholds preserve two generic partitions.
            // Headcount bounds both lists; an edge can replace at most two.
            if loss.is_finite() && row.improvement > (slot + 1) as f64 * loss {
                rows.retain(|r| !overlaps(&r.heads, &row.heads));
                rows.push(row.clone());
            }
        }
    }
}

fn pair_replay(
    cuts: &HullRootMipCuts<'_>,
    row: &joint::JointMajorant,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let (weights, g) = joint::proposal(cuts, &row.heads)?;
    if !same_slice(&weights, &row.weights) || !same_slice(&g, &row.g) {
        return None;
    }
    let upper = joint::support(cuts.problem, &row.heads, &weights, &g, end, poll)?;
    let gain = joint::improvement(cuts, &row.heads, &weights, &g, upper, end, poll)?;
    (same(upper, row.upper) && same(gain, row.improvement) && gain > 0.0).then_some(())
}

struct Sum {
    point: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    upper: f64,
}
impl Sum {
    fn new(n: usize) -> Self {
        Self {
            point: vec![0.0; n],
            lo: vec![0.0; n],
            hi: vec![0.0; n],
            upper: 0.0,
        }
    }
    fn add(
        &mut self,
        g: &[f64],
        upper: f64,
        weight: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        if g.len() != self.point.len() || !weight.is_finite() || weight <= 0.0 {
            return None;
        }
        self.upper = finite(add_up(self.upper, mul_up(weight, upper)))?;
        for (i, &v) in g.iter().enumerate() {
            if !live(end) || !poll() {
                return None;
            }
            self.point[i] = finite(self.point[i] + weight * v)?;
            self.lo[i] = finite(add_down(self.lo[i], mul_down(weight, v)))?;
            self.hi[i] = finite(add_up(self.hi[i], mul_up(weight, v)))?;
        }
        Some(())
    }
    fn finish(
        mut self,
        p: &HullMarginProblem,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<(Vec<f64>, f64)> {
        // A.r-G.x = sum(component LHS)+(G_exact-G).x. Price every
        // input residual, including singleton products and sum rounding.
        for i in 0..p.input_dim {
            if !live(end) || !poll() {
                return None;
            }
            let lo = finite(sub_down(self.lo[i], self.point[i]))?;
            let hi = finite(sub_up(self.hi[i], self.point[i]))?;
            let (l, u) = (p.lower[i], p.upper[i]);
            let extra = mul_up(lo, l)
                .max(mul_up(lo, u))
                .max(mul_up(hi, l))
                .max(mul_up(hi, u));
            self.upper = finite(add_up(self.upper, finite(extra)?))?;
        }
        Some((self.point, self.upper))
    }
}

fn checked_planes(
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let p = cuts.problem;
    let negative: Vec<_> =
        p.c.iter()
            .enumerate()
            .filter_map(|(j, c)| (*c < 0.0).then_some(j))
            .collect();
    if negative.len() != cuts.planes.len() {
        return None;
    }
    for (&head, plane) in negative.iter().zip(&cuts.planes) {
        if !live(end)
            || !poll()
            || head != plane.head
            || plane.g.len() != p.input_dim
            || !same_slice(
                &plane.g,
                cuts.multipliers
                    .g
                    .get(head * p.input_dim..(head + 1) * p.input_dim)?,
            )
        {
            return None;
        }
        let upper = majorant_offset_upper(p.w_row(head), p.b1[head], &plane.g, &p.lower, &p.upper)?;
        if !same(upper, plane.upper) {
            return None;
        }
    }
    Some(())
}

fn uniform_gain(
    cuts: &HullRootMipCuts<'_>,
    g: &[f64],
    upper: f64,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let p = cuts.problem;
    let mut lo = vec![0.0; p.input_dim];
    let mut hi = lo.clone();
    let mut gain = 0.0;
    for plane in &cuts.planes {
        if !live(end) || !poll() {
            return None;
        }
        let a = -p.c[plane.head];
        gain = finite(add_down(gain, mul_down(a, plane.upper)))?;
        for i in 0..p.input_dim {
            if !live(end) || !poll() {
                return None;
            }
            lo[i] = finite(add_down(lo[i], mul_down(a, plane.g[i])))?;
            hi[i] = finite(add_up(hi[i], mul_up(a, plane.g[i])))?;
        }
    }
    for i in 0..p.input_dim {
        if !live(end) || !poll() {
            return None;
        }
        let l = finite(sub_down(lo[i], g[i]))?;
        let u = finite(sub_up(hi[i], g[i]))?;
        let corner = mul_down(l, p.lower[i])
            .min(mul_down(l, p.upper[i]))
            .min(mul_down(u, p.lower[i]))
            .min(mul_down(u, p.upper[i]));
        gain = finite(add_down(gain, finite(corner)?))?;
    }
    finite(sub_down(gain, upper))
}

fn build(
    cuts: &HullRootMipCuts<'_>,
    mut pairs: Vec<joint::JointMajorant>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Aggregate> {
    let p = cuts.problem;
    if !capacity(p)
        || !live(end)
        || !poll()
        || pairs.is_empty()
        || pairs.len() > p.hidden_dim / 2
        || pairs.len() > cuts.joint_completed
        || cuts.joint_attempted > joint::MAX_PAIRS
        || cuts.joint_completed > cuts.joint_attempted
    {
        return None;
    }
    require_hull_environment().ok()?;
    checked_planes(cuts, end, poll)?;
    pairs.sort_by_key(|r| r.heads);
    let mut used = vec![false; p.hidden_dim];
    let mut sum = Sum::new(p.input_dim);
    for row in &pairs {
        pair_replay(cuts, row, end, poll)?;
        for &j in &row.heads {
            if j >= used.len() || used[j] {
                return None;
            }
            used[j] = true;
        }
        sum.add(&row.g, row.upper, 1.0, end, poll)?;
    }
    for plane in &cuts.planes {
        if !used[plane.head] {
            sum.add(&plane.g, plane.upper, -p.c[plane.head], end, poll)?;
        }
    }
    let (g, upper) = sum.finish(p, end, poll)?;
    let improvement = uniform_gain(cuts, &g, upper, end, poll)?;
    if improvement <= 0.0 || !live(end) || !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    Some(Aggregate {
        groups: Vec::new(),
        group_budget: None,
        _group_memory: None,
        pairs,
        g,
        upper,
        improvement,
    })
}

pub(super) fn finish(
    cuts: &HullRootMipCuts<'_>,
    partitions: Partitions,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Aggregate> {
    let mut best: Option<Aggregate> = None;
    for pairs in partitions.candidates {
        if let Some(row) = build(cuts, pairs, end, poll) {
            if best
                .as_ref()
                .is_none_or(|b| row.improvement > b.improvement)
            {
                best = Some(row);
            }
        }
        if !live(end) || !poll() {
            break;
        }
    }
    best
}

pub(super) fn replay(
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Aggregate> {
    let old = cuts.compact.as_ref()?;
    let row = if old.groups.is_empty() {
        build(cuts, old.pairs.clone(), end, poll)?
    } else {
        let budget = old.group_budget.as_ref()?;
        let mut rows = Vec::new();
        for row in &old.groups {
            rows.push(group::replay(cuts, row, budget, poll)?);
        }
        group_rows::build_groups(cuts, rows, budget, poll, true)?
    };
    (same_slice(&row.g, &old.g)
        && same(row.upper, old.upper)
        && same(row.improvement, old.improvement))
    .then_some(row)
}

/// A separate, freshly replayed FULL original objective. Never add a search
/// gain to the LP-attained scalar or assert that its old tuple attains this.
pub(super) fn root_lower(
    cuts: &HullRootMipCuts<'_>,
    row: &Aggregate,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    require_hull_environment().ok()?;
    let p = cuts.problem;
    if row.g.len() != p.input_dim || cuts.multipliers.alpha.len() != p.hidden_dim {
        return None;
    }
    let mut lo: Vec<_> = row.g.iter().map(|v| -v).collect();
    let mut hi = lo.clone();
    let mut constant = finite(sub_down(p.d, row.upper))?;
    for j in 0..p.hidden_dim {
        if !live(end) || !poll() {
            return None;
        }
        if p.c[j] <= 0.0 {
            continue;
        }
        let alpha = sanitize_alpha(cuts.multipliers.alpha[j]);
        if alpha == 0.0 {
            continue;
        }
        let (kl, ku) = (mul_down(p.c[j], alpha), mul_up(p.c[j], alpha));
        constant = finite(add_down(constant, scale_interval(kl, ku, p.b1[j]).0))?;
        for i in 0..p.input_dim {
            if !live(end) || !poll() {
                return None;
            }
            let (l, u) = scale_interval(kl, ku, p.w_row(j)[i]);
            lo[i] = finite(add_down(lo[i], l))?;
            hi[i] = finite(add_up(hi[i], u))?;
        }
    }
    for i in 0..p.input_dim {
        if !live(end) || !poll() {
            return None;
        }
        let term = mul_down(lo[i], p.lower[i])
            .min(mul_down(lo[i], p.upper[i]))
            .min(mul_down(hi[i], p.lower[i]))
            .min(mul_down(hi[i], p.upper[i]));
        constant = finite(add_down(constant, finite(term)?))?;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(constant)
}

pub(super) fn prepare_rows(
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    columns: &[IrCol],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    if cuts.compact.is_none() {
        return Some(Vec::new());
    }
    let row = replay(cuts, end, poll)?;
    let p = cuts.problem;
    let mut coeffs = Vec::with_capacity(cuts.planes.len() + p.input_dim);
    for plane in &cuts.planes {
        if !live(end) || !poll() {
            return None;
        }
        coeffs.push((columns.get(plane.head)?.0, -p.c[plane.head]));
    }
    for (i, &g) in row.g.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        if g != 0.0 {
            coeffs.push((parts.input_vars.get(i)?.0, -g));
        }
    }
    (live(end) && poll()).then_some(vec![RowSpec {
        lb: f64::NEG_INFINITY,
        ub: row.upper,
        coeffs,
    }])
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_compact_tests.rs"]
mod tests;

#[path = "single_hidden_hull_mip_compact_groups.rs"]
mod group_rows;
pub(super) use group_rows::build_group_aggregate;
