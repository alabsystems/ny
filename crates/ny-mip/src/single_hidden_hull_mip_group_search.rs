// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Neutral geometric growth is proposal search only: forest prefixes may
//! have zero gain even when their completed odd cycle has a strict gain.
use super::*;
use std::collections::BTreeSet;
const MAX_EDGES: usize = 2048;

#[derive(Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
    score: f64,
}
fn edges(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize],
    b: &Budget,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Vec<Edge>, Permit)> {
    let p = cuts.problem;
    let n = heads.len();
    let memory = b.memory(MAX_EDGES * 32 + p.hidden_dim * 48 + MAX_ATTEMPTS * 256)?;
    b.charge(n.checked_mul(p.input_dim + 1)?.checked_mul(3)?, 0, false)?;
    let mut norms = vec![0.; n];
    for (i, &j) in heads.iter().enumerate() {
        if !live(b.search_end) || !poll() {
            return None;
        }
        norms[i] = p.w_row(j).iter().map(|x| x * x).sum::<f64>();
    }
    let mut out = Vec::with_capacity(MAX_EDGES);
    // Gap order spreads a bounded scan across the full head list. The cap
    // applies to actual coordinate-scored pairs, not merely retained edges.
    'scan: for gap in 1..n {
        for a in 0..n - gap {
            if out.len() == MAX_EDGES {
                break 'scan;
            }
            b.charge(3 * (p.input_dim + 1), 0, false)?;
            if !poll() {
                return None;
            }
            let bb = a + gap;
            let dot: f64 = p
                .w_row(heads[a])
                .iter()
                .zip(p.w_row(heads[bb]))
                .map(|(x, y)| x * y)
                .sum();
            let den = (norms[a] * norms[bb]).sqrt();
            let score = if den > 0. { dot.abs() / den } else { 0. };
            out.push(Edge {
                a: heads[a],
                b: heads[bb],
                score: if score.is_finite() { score } else { 0. },
            });
        }
    }
    Some((out, memory))
}
fn grow(
    seed: usize,
    width: usize,
    edges: &[Edge],
    blocked: &[bool],
    partners: &[Option<usize>],
    b: &Budget,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<usize>> {
    let mut group = vec![seed];
    let mut scores = vec![0.; width];
    let mut last = seed;
    while group.len() < MAX_GROUP {
        // Incremental scalar edge scores: one bounded edge scan per added
        // head, no fresh d-loop, and no per-member neighbour expansion.
        b.charge(
            edges.len().checked_mul(3)?.checked_add(width * 4)?,
            0,
            false,
        )?;
        if !poll() {
            return None;
        }
        for e in edges {
            if e.a == last {
                scores[e.b] += e.score;
            }
            if e.b == last {
                scores[e.a] += e.score;
            }
        }
        // Keep a completed baseline pair together whenever space permits.
        let paired = partners[last].filter(|j| !blocked[*j] && !group.contains(j));
        let next = paired.or_else(|| {
            (0..width)
                .filter(|i| !blocked[*i] && !group.contains(i) && scores[*i] > 0.)
                .max_by(|a, bb| scores[*a].total_cmp(&scores[*bb]).then(bb.cmp(a)))
        });
        let Some(next) = next else { break };
        group.push(next);
        last = next;
    }
    group.sort_unstable();
    Some(group)
}
fn consider(
    cuts: &mut HullRootMipCuts<'_>,
    rows: Vec<GroupMajorant>,
    b: &Budget,
    best: &mut f64,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let aggregate = compact::build_group_aggregate(cuts, rows, b, poll)?;
    let floor = compact::root_lower(cuts, &aggregate, b.search_end, poll)?;
    if floor > *best {
        let (work, masks) = replay_cost(cuts, &aggregate.groups)?;
        b.reserve(work, masks)?;
        *best = floor;
        cuts.compact = Some(aggregate);
        cuts.compact_mode = true;
    }
    Some(())
}
pub(in crate::single_hidden_hull::mip_cuts) fn refine(
    cuts: &mut HullRootMipCuts<'_>,
    authority: Instant,
    reserve: Duration,
) {
    if let Some(budget) = Budget::new(authority.min(cuts.deadline), reserve) {
        refine_with_budget(cuts, &budget);
    }
}
pub(in crate::single_hidden_hull::mip_cuts) fn refine_with_budget(
    cuts: &mut HullRootMipCuts<'_>,
    budget: &Budget,
) {
    refine_with_budget_and_poll(cuts, budget, &mut || true);
}
pub(in crate::single_hidden_hull::mip_cuts) fn refine_with_budget_and_poll(
    cuts: &mut HullRootMipCuts<'_>,
    b: &Budget,
    poll: &mut dyn FnMut() -> bool,
) {
    // Optional failure never replaces or erases any completed old geometry.
    if !live(b.search_end)
        || !live(cuts.deadline)
        || !poll()
        || require_hull_environment().is_err()
        || cuts.group_budget.is_some()
    {
        return;
    }
    cuts.group_budget = Some(b.clone());
    let p = cuts.problem;
    let terms = (p.hidden_dim + 1).saturating_mul(p.input_dim + 1);
    if b.charge(4 * terms, 0, false).is_none() {
        return;
    }
    let mut best = cuts.checked_lower;
    if let Some(row) = cuts.compact.as_ref() {
        let Some(floor) = compact::root_lower(cuts, row, b.search_end, poll) else {
            return;
        };
        best = best.max(floor);
    }
    if best > 0. {
        return;
    }
    let mut heads: Vec<_> = cuts.planes.iter().map(|r| r.head).collect();
    if b.charge(8 * terms, 0, false).is_none() {
        return;
    }
    let mut priority = vec![0.; p.hidden_dim];
    for &j in &heads {
        let start = j * p.input_dim;
        priority[j] = -p.c[j]
            * (0..p.input_dim)
                .map(|i| {
                    let g = cuts.multipliers.g[start + i];
                    ((p.w_row(j)[i] - g).abs() + g.abs()) * (p.upper[i] - p.lower[i])
                })
                .sum::<f64>();
        if !priority[j].is_finite() {
            priority[j] = 0.;
        }
    }
    heads.sort_by(|a, bb| priority[*bb].total_cmp(&priority[*a]).then(a.cmp(bb)));
    let Some((edges, _memory)) = edges(cuts, &heads, b, poll) else {
        return;
    };
    let mut partners = vec![None; p.hidden_dim];
    if let Some(old) = cuts.compact.as_ref() {
        for row in &old.pairs {
            partners[row.heads[0]] = Some(row.heads[1]);
            partners[row.heads[1]] = Some(row.heads[0]);
        }
    }
    let mut seen = BTreeSet::new();
    let mut planned = Vec::<GroupMajorant>::new();
    let mut used = vec![false; p.hidden_dim];
    // First cover previously unplanned negative mass, including neutral
    // intermediate certificates. Only stronger completed aggregates commit.
    for &seed in &heads {
        if used[seed] {
            continue;
        }
        let Some(group) = grow(seed, p.hidden_dim, &edges, &used, &partners, b, poll) else {
            return;
        };
        if group.len() < 2 {
            used[seed] = true;
            continue;
        }
        if !seen.insert(group.clone()) {
            continue;
        }
        let Some(row) = support(cuts, &group, b, poll) else {
            return;
        };
        for &j in row.heads() {
            used[j] = true;
        }
        planned.push(row);
        if consider(cuts, planned.clone(), b, &mut best, poll).is_none() {
            return;
        }
        if best > 0. {
            return;
        }
    }
    // A second bounded track may replace overlapping groups after the first
    // disjoint track; it never resets a clock, counter, or publication reserve.
    let blocked = vec![false; p.hidden_dim];
    for &seed in heads.iter().take(MAX_ATTEMPTS) {
        let Some(group) = grow(seed, p.hidden_dim, &edges, &blocked, &partners, b, poll) else {
            break;
        };
        if group.len() < 2 || !seen.insert(group.clone()) {
            continue;
        }
        let Some(row) = support(cuts, &group, b, poll) else {
            break;
        };
        let mut rows = cuts.compact.as_ref().map_or_else(Vec::new, |a| {
            a.groups
                .iter()
                .filter(|old| !old.heads().iter().any(|j| row.heads().contains(j)))
                .cloned()
                .collect()
        });
        rows.push(row);
        if consider(cuts, rows, b, &mut best, poll).is_none() || best > 0. {
            break;
        }
    }
}
