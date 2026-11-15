// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Nominal equality-simplex block minimizer. This grants no certificate authority.
use super::*;

#[cfg(test)]
#[path = "input_head_directional_allocation_projection_tests.rs"]
mod tests;

pub(super) struct Update {
    pub(super) masses: Vec<f64>,
    pub(super) residual: Vec<Vec<f64>>,
    pub(super) changed: bool,
}

/// One k*d dot pass, one checked k*d new-residual pass, bounded sort and
/// projection bookkeeping. The caller charges this before allocating.
pub(super) fn cost(k: usize, d: usize) -> Option<usize> {
    if !(3..=MAX_CUTS * MAX_ROUNDS).contains(&k) || d == 0 {
        return None;
    }
    let levels = usize::BITS as usize - (k - 1).leading_zeros() as usize;
    k.checked_mul(d)?
        .checked_mul(2)?
        .checked_add(k.checked_mul(levels)?.checked_mul(3)?)?
        .checked_add(k.checked_mul(8)?)?
        .checked_add(16)
}

/// Bottom-up mergesort: each level uses <=k comparisons and <=k writes;
/// initial indices/scratch and ties are covered by the separate O(k) charge.
fn descending(z: &[f64]) -> Option<Vec<usize>> {
    let k = z.len();
    if k == 0 || k > MAX_CUTS * MAX_ROUNDS || z.iter().any(|&v| !good(v)) {
        return None;
    }
    let mut order: Vec<_> = (0..k).collect();
    let mut scratch = vec![0usize; k];
    let mut width = 1usize;
    while width < k {
        let stride = width.checked_mul(2)?;
        for start in (0..k).step_by(stride) {
            let middle = start.checked_add(width)?.min(k);
            let end = start.checked_add(stride)?.min(k);
            let (mut left, mut right) = (start, middle);
            for slot in &mut scratch[start..end] {
                let take_left = right == end
                    || (left < middle
                        && (z[order[left]] > z[order[right]]
                            || (z[order[left]] == z[order[right]] && order[left] < order[right])));
                if take_left {
                    *slot = order[left];
                    left += 1;
                } else {
                    *slot = order[right];
                    right += 1;
                }
            }
        }
        std::mem::swap(&mut order, &mut scratch);
        width = stride;
    }
    Some(order)
}

/// In real arithmetic this is Proj_{a>=0,sum a=mass}(z). Represented outputs
/// still need existing outward normalization and independent signed proof.
pub(super) fn project(z: &[f64], mass: f64) -> Option<Vec<f64>> {
    if !good(mass) || mass <= 0.0 {
        return None;
    }
    let order = descending(z)?;
    let (mut total, mut threshold) = (0.0, None);
    for (at, &g) in order.iter().enumerate() {
        total += z[g];
        let trial = (total - mass) / (at + 1) as f64;
        if !good(total) || !good(trial) {
            return None;
        }
        if z[g] > trial {
            threshold = Some(trial);
        }
    }
    let threshold = threshold?;
    z.iter()
        .map(|&v| {
            let a = (v - threshold).max(0.0);
            (good(a) && a >= 0.0).then_some(a)
        })
        .collect()
}

/// All outputs are local until the caller atomically replaces the coordinate.
/// Never clone then update/validate an entire residual matrix outside its
/// charged pass. A failure at the final entry leaves both inputs untouched.
pub(super) fn coordinate(col: &Column, old: &[f64], residual: &[Vec<f64>]) -> Option<Update> {
    let (k, d) = (old.len(), col.values.len());
    if !(3..=MAX_CUTS * MAX_ROUNDS).contains(&k)
        || d == 0
        || residual.len() != k
        || !good(col.norm)
        || col.norm <= 0.0
        || !good(col.mass)
        || col.mass <= 0.0
        || old.iter().any(|&a| !good(a) || a < 0.0)
    {
        return None;
    }
    let mut z = Vec::with_capacity(k);
    for (g, row) in residual.iter().enumerate() {
        if row.len() != d {
            return None;
        }
        let mut dot = 0.0;
        for (&w, &r) in col.values.iter().zip(row) {
            if !good(w) || !good(r) {
                return None;
            }
            dot += w * r;
        }
        let value = old[g] + dot / col.norm;
        if !good(dot) || !good(value) {
            return None;
        }
        z.push(value);
    }
    let masses = project(&z, col.mass)?;
    let mut next = Vec::with_capacity(k);
    let mut changed = false;
    for (g, row) in residual.iter().enumerate() {
        let delta = masses[g] - old[g];
        if !good(delta) {
            return None;
        }
        changed |= delta != 0.0;
        let mut updated = Vec::with_capacity(d);
        for (&r, &w) in row.iter().zip(&col.values) {
            let value = r - delta * w;
            if !good(value) {
                return None;
            }
            updated.push(value);
        }
        next.push(updated);
    }
    Some(Update {
        masses,
        residual: next,
        changed,
    })
}
