// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint upper planes for two negative heads. Search is advice; every retained
//! row is replayed over every algebraic support mask and original input.

use super::*;
use crate::ir::Col as IrCol;

pub(super) const MAX_PAIRS: usize = 2_048;
const MAX_ROWS: usize = 64;
const REPLAY_RESERVE: Duration = Duration::from_millis(75);

#[derive(Debug, Clone)]
pub(super) struct JointMajorant {
    pub(super) heads: [usize; 2],
    pub(super) weights: [f64; 2],
    pub(super) g: Vec<f64>,
    pub(super) upper: f64,
    pub(super) improvement: f64,
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}
fn same_slice(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(*a, *b))
}

/// For positive a, sum a_j ReLU(t_j) is exactly the maximum of the four
/// affine support branches, including the empty branch. Each coefficient is
/// carried outward before all four interval/box corners are priced.
pub(super) fn support(
    p: &HullMarginProblem,
    heads: &[usize; 2],
    weights: &[f64; 2],
    g: &[f64],
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    if !capacity(p)
        || !live(deadline)
        || !poll()
        || heads[0] >= heads[1]
        || heads[1] >= p.hidden_dim
        || weights.iter().any(|a| !a.is_finite() || *a <= 0.0)
        || g.len() != p.input_dim
        || g.iter().any(|v| !v.is_finite())
        || p.w1.len() != p.hidden_dim.checked_mul(p.input_dim)?
        || p.b1.len() != p.hidden_dim
        || p.lower.len() != p.input_dim
        || p.upper.len() != p.input_dim
    {
        return None;
    }
    require_hull_environment().ok()?;
    let mut upper = f64::NEG_INFINITY;
    for mask in 0..4 {
        if !live(deadline) || !poll() {
            return None;
        }
        let mut value = 0.0;
        for j in 0..2 {
            if mask & (1 << j) != 0 {
                value = finite(add_up(value, mul_up(weights[j], p.b1[heads[j]])))?;
            }
        }
        for i in 0..p.input_dim {
            if !live(deadline) || !poll() {
                return None;
            }
            let (l, u) = (p.lower[i], p.upper[i]);
            if !l.is_finite() || !u.is_finite() || l > u {
                return None;
            }
            let (mut lo, mut hi) = (-g[i], -g[i]);
            for j in 0..2 {
                if mask & (1 << j) != 0 {
                    let w = p.w_row(heads[j])[i];
                    if !w.is_finite() {
                        return None;
                    }
                    lo = finite(add_down(lo, mul_down(weights[j], w)))?;
                    hi = finite(add_up(hi, mul_up(weights[j], w)))?;
                }
            }
            let corner = mul_up(lo, l)
                .max(mul_up(lo, u))
                .max(mul_up(hi, l))
                .max(mul_up(hi, u));
            value = finite(add_up(value, finite(corner)?))?;
        }
        upper = upper.max(value);
    }
    if !live(deadline) || !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    finite(upper)
}

fn planes<'a>(cuts: &'a HullRootMipCuts<'_>, heads: &[usize; 2]) -> Option<[&'a Majorant; 2]> {
    if heads[0] >= heads[1] {
        return None;
    }
    Some([
        cuts.planes.iter().find(|p| p.head == heads[0])?,
        cuts.planes.iter().find(|p| p.head == heads[1])?,
    ])
}

/// The rounded sum is just a finite point proposal. Its rounding is never
/// treated as conservation of the original objective or individual rows.
pub(super) fn proposal(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize; 2],
) -> Option<([f64; 2], Vec<f64>)> {
    let p = cuts.problem;
    let pair = planes(cuts, heads)?;
    let weights = [-*p.c.get(heads[0])?, -*p.c.get(heads[1])?];
    if weights.iter().any(|a| !a.is_finite() || *a <= 0.0)
        || pair.iter().any(|r| r.g.len() != p.input_dim)
    {
        return None;
    }
    let mut g = Vec::with_capacity(p.input_dim);
    for i in 0..p.input_dim {
        g.push(finite(
            weights[0] * pair[0].g[i] + weights[1] * pair[1].g[i],
        )?);
    }
    Some((weights, g))
}

/// A LOWER bound on the improvement over the exact weighted sum of the
/// already-exported individual rows, uniformly on the whole original box.
/// Price the complete coefficient residual caused by the point sum g. This
/// score is selection metadata, never a new objective certificate.
pub(super) fn improvement(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize; 2],
    weights: &[f64; 2],
    g: &[f64],
    upper: f64,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let pair = planes(cuts, heads)?;
    let p = cuts.problem;
    if g.len() != p.input_dim {
        return None;
    }
    let mut lower = finite(add_down(
        mul_down(weights[0], pair[0].upper),
        mul_down(weights[1], pair[1].upper),
    ))?;
    for i in 0..p.input_dim {
        if !live(deadline) || !poll() {
            return None;
        }
        let lo = finite(sub_down(
            add_down(
                mul_down(weights[0], pair[0].g[i]),
                mul_down(weights[1], pair[1].g[i]),
            ),
            g[i],
        ))?;
        let hi = finite(sub_up(
            add_up(
                mul_up(weights[0], pair[0].g[i]),
                mul_up(weights[1], pair[1].g[i]),
            ),
            g[i],
        ))?;
        let (l, u) = (p.lower[i], p.upper[i]);
        let corner = mul_down(lo, l)
            .min(mul_down(lo, u))
            .min(mul_down(hi, l))
            .min(mul_down(hi, u));
        lower = finite(add_down(lower, finite(corner)?))?;
    }
    finite(sub_down(lower, upper))
}

fn priority(cuts: &HullRootMipCuts<'_>, head: usize) -> f64 {
    let p = cuts.problem;
    let mut width = 0.0;
    for i in 0..p.input_dim {
        width += p.w_row(head)[i].abs() * (p.upper[i] - p.lower[i]);
    }
    let score = -p.c[head] * width;
    if score.is_nan() {
        0.0
    } else {
        score
    }
}

/// One optional pass, bounded by pair/row counts and a subordinate search
/// cutoff. Reserve replay time INSIDE the same stored original stage clock.
/// Any incomplete later pair is discarded; completed individual/joint rows
/// remain immutable. Larger head sets intentionally receive a partial pool.
pub(super) fn discover_compact(
    cuts: &mut HullRootMipCuts<'_>,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) {
    discover_impl(cuts, deadline, poll, true);
}

fn discover_impl(
    cuts: &mut HullRootMipCuts<'_>,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
    compact_mode: bool,
) {
    if cuts.joint_searched {
        return;
    }
    cuts.joint_searched = true;
    cuts.compact_mode = compact_mode;
    let mut partitions = compact_mode.then(compact::Partitions::new);
    let deadline = deadline.min(cuts.deadline);
    let deadline = ny_core::instance_budget::deadline().map_or(deadline, |d| d.min(deadline));
    let Some(stop) = deadline.checked_sub(REPLAY_RESERVE) else {
        return;
    };
    if !live(stop) || !poll() || require_hull_environment().is_err() {
        return;
    }
    let mut order: Vec<_> = cuts
        .planes
        .iter()
        .map(|r| (r.head, priority(cuts, r.head)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    'pairs: for i in 0..order.len() {
        for j in i + 1..order.len() {
            if cuts.joint_attempted >= MAX_PAIRS || !live(stop) || !poll() {
                break 'pairs;
            }
            cuts.joint_attempted += 1;
            let mut heads = [order[i].0, order[j].0];
            heads.sort_unstable();
            let Some((weights, g)) = proposal(cuts, &heads) else {
                continue;
            };
            let Some(upper) = support(cuts.problem, &heads, &weights, &g, stop, poll) else {
                continue;
            };
            let Some(improvement) = improvement(cuts, &heads, &weights, &g, upper, stop, poll)
            else {
                continue;
            };
            cuts.joint_completed += 1;
            if improvement <= 0.0 {
                continue;
            }
            let row = JointMajorant {
                heads,
                weights,
                g,
                upper,
                improvement,
            };
            if let Some(partitions) = &mut partitions {
                partitions.observe(&row);
            }
            cuts.joint.push(row);
            cuts.joint.sort_by(|a, b| {
                b.improvement
                    .total_cmp(&a.improvement)
                    .then_with(|| a.heads.cmp(&b.heads))
            });
            cuts.joint.truncate(MAX_ROWS);
        }
    }
    // Both partitions use completed proposals from this one bounded search.
    // A failed optional build retains every individual row; no64-row fallback.
    if let Some(partitions) = partitions {
        cuts.compact = compact::finish(cuts, partitions, stop, poll);
    }
}

/// Called only after the original whole-model and individual-plane binding.
/// Replay every four-mask proof and exact tuple seal before any IR append.
pub(super) fn prepare_rows(
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    columns: &[IrCol],
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    if cuts.joint.len() > MAX_ROWS
        || cuts.joint_attempted > MAX_PAIRS
        || cuts.joint_completed > cuts.joint_attempted
        || cuts.joint.len() > cuts.joint_completed
    {
        return None;
    }
    let mut rows = Vec::with_capacity(cuts.joint.len());
    for (index, row) in cuts.joint.iter().enumerate() {
        if !live(deadline)
            || !poll()
            || cuts.joint[..index].iter().any(|r| r.heads == row.heads)
            || !row.improvement.is_finite()
            || row.improvement <= 0.0
        {
            return None;
        }
        let (weights, g) = proposal(cuts, &row.heads)?;
        if !same_slice(&weights, &row.weights) || !same_slice(&g, &row.g) {
            return None;
        }
        let upper = support(cuts.problem, &row.heads, &weights, &g, deadline, poll)?;
        let gain = improvement(cuts, &row.heads, &weights, &g, upper, deadline, poll)?;
        if !same(upper, row.upper) || !same(gain, row.improvement) || gain <= 0.0 {
            return None;
        }
        let mut coeffs = Vec::with_capacity(cuts.problem.input_dim + 2);
        for j in 0..2 {
            coeffs.push((columns.get(row.heads[j])?.0, weights[j]));
        }
        for (i, &coefficient) in g.iter().enumerate() {
            if !poll() {
                return None;
            }
            if coefficient != 0.0 {
                coeffs.push((parts.input_vars.get(i)?.0, -coefficient));
            }
        }
        rows.push(RowSpec {
            lb: f64::NEG_INFINITY,
            ub: upper,
            coeffs,
        });
    }
    (live(deadline) && poll()).then_some(rows)
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_joint_tests.rs"]
mod tests;
