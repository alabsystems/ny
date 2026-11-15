// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded allocation proposals; represented outputs are independently re-certified.
use super::*;

#[path = "input_head_directional_allocation_projection.rs"]
mod projection;

#[cfg(test)]
#[path = "input_head_directional_allocation_source_tests.rs"]
mod source_tests;

pub(super) fn fit(
    negative: &[Column],
    groups: &[Vec<usize>],
    positive: &[Column],
    work: &mut Search<'_>,
) -> Option<Vec<Vec<f64>>> {
    let (k, d) = (groups.len(), positive.first()?.values.len());
    work.charge(
        k.checked_mul(d)?
            .checked_add(positive.len().checked_mul(k)?)?,
    )?;
    let mut residual = vec![vec![0.0; d]; k];
    let mut group_mass = vec![0.0; k];
    for (g, indices) in groups.iter().enumerate() {
        for &at in indices {
            let col = negative.get(at)?;
            work.charge(d.checked_add(1)?)?;
            group_mass[g] += col.mass;
            for (r, &v) in residual[g].iter_mut().zip(&col.values) {
                *r += col.mass * v;
            }
        }
    }
    let total: f64 = group_mass.iter().sum();
    if !good(total) || total <= 0.0 {
        return None;
    }
    let mut masses = vec![vec![0.0; k]; positive.len()];
    for (p, col) in positive.iter().enumerate() {
        for g in 0..k {
            work.charge(d.checked_add(2)?)?;
            let a = col.mass * (group_mass[g] / total);
            if !good(a) || a < 0.0 {
                return None;
            }
            masses[p][g] = a;
            for (r, &v) in residual[g].iter_mut().zip(&col.values) {
                *r -= a * v;
            }
        }
    }
    // A failed optional coordinate leaves the last complete allocation intact.
    let tail = positive
        .len()
        .checked_mul(k.checked_mul(7)?.checked_add(4)?)?
        .checked_add(negative.len().checked_mul(4)?)?
        .checked_add(32)?;
    improve(positive, &mut masses, &mut residual, work, tail)?;
    // Search conserves mass only over reals. Round down represented masses,
    // then independently check their outward total against original q.lo.
    for (p, col) in positive.iter().enumerate() {
        work.charge(k.checked_mul(6)?.checked_add(4)?)?;
        let sum = masses[p]
            .iter()
            .try_fold(I::p(0.0), |s, &a| s.add(I::p(a)))?;
        if !sum.valid() || sum.hi < 0.0 {
            return None;
        }
        let scale = if sum.hi > 0.0 {
            (col.mass / sum.hi).min(1.0).next_down().max(0.0)
        } else {
            0.0
        };
        for a in &mut masses[p] {
            *a = (*a * scale).next_down().max(0.0);
        }
        let sum = masses[p]
            .iter()
            .try_fold(I::p(0.0), |s, &a| s.add(I::p(a)))?;
        if sum.hi > col.mass {
            return None;
        }
    }
    Some(masses)
}

fn improve(
    positive: &[Column],
    masses: &mut [Vec<f64>],
    residual: &mut Vec<Vec<f64>>,
    work: &mut Search<'_>,
    tail: usize,
) -> Option<()> {
    let (k, d) = (residual.len(), positive.first()?.values.len());
    'sweeps: for _ in 0..8 {
        let mut changed = false;
        for (p, col) in positive.iter().enumerate() {
            let cost = if k >= 3 {
                projection::cost(k, d)?
            } else {
                k.checked_add(2)?
                    .checked_mul(d)?
                    .checked_add(k.checked_mul(3)?)?
            };
            if !work.room(cost.checked_add(tail)?) {
                break 'sweeps;
            }
            if work.charge(cost).is_none() {
                break 'sweeps;
            }
            if col.norm <= 0.0 || !good(col.norm) {
                continue;
            }
            if k >= 3 {
                // Prepare every represented mass/residual before mutation.
                // Optional failure preserves the last COMPLETE allocation.
                let Some(update) = projection::coordinate(col, &masses[p], residual) else {
                    break 'sweeps;
                };
                if update.changed {
                    masses[p] = update.masses;
                    *residual = update.residual;
                    changed = true;
                }
                continue;
            }
            // With two groups this original pair step is already the exact
            // real equality-simplex coordinate minimizer; keep its old costs.
            let dots: Vec<f64> = residual
                .iter()
                .map(|r| col.values.iter().zip(r).map(|(w, r)| w * r).sum())
                .collect();
            if dots.iter().any(|&v| !good(v)) {
                return None;
            }
            let g = (0..k).max_by(|&a, &b| dots[a].total_cmp(&dots[b]).then(b.cmp(&a)))?;
            let Some(h) = (0..k)
                .filter(|&g| masses[p][g] > 0.0)
                .min_by(|&a, &b| dots[a].total_cmp(&dots[b]).then(a.cmp(&b)))
            else {
                continue;
            };
            if g == h {
                continue;
            }
            let delta = ((dots[g] - dots[h]) / (2.0 * col.norm)).clamp(-masses[p][g], masses[p][h]);
            let (to, from) = (masses[p][g] + delta, masses[p][h] - delta);
            if !good(delta) || !good(to) || !good(from) || to < 0.0 || from < 0.0 {
                return None;
            }
            if delta == 0.0 {
                continue;
            }
            for j in 0..d {
                residual[g][j] -= delta * col.values[j];
                residual[h][j] += delta * col.values[j];
                if !good(residual[g][j]) || !good(residual[h][j]) {
                    return None;
                }
            }
            masses[p][g] = to;
            masses[p][h] = from;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    Some(())
}
