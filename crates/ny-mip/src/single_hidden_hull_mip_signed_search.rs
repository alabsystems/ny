// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Two finite coordinate sweeps are untrusted steering, never certificates.
use super::*;
#[path = "single_hidden_hull_mip_signed_alpha.rs"]
pub(super) mod alpha;

pub(super) fn legacy_work(d: usize) -> Option<usize> {
    d.checked_add(4)?
        .checked_mul(d.checked_add(1)?)?
        .checked_mul(8)
}
pub(super) fn coordinate_work(d: usize) -> Option<usize> {
    Some(legacy_work(d)?.min(alpha::work(d)?))
}
pub(super) fn scratch_bytes(d: usize) -> Option<usize> {
    if alpha::work(d)? < legacy_work(d)? {
        32usize.checked_mul(d.checked_add(1)?)
    } else {
        Some(0)
    }
}
fn score(p: &HullMarginProblem, q: &[f64], constant: f64) -> Option<f64> {
    let mut value = constant;
    for i in 0..p.input_dim {
        value = finite(value + (q[i] * p.lower[i]).min(q[i] * p.upper[i]))?;
    }
    Some(value)
}
pub(super) fn mask_cache(
    p: &HullMarginProblem,
    c: &Component,
    g: &[f64],
    mask: usize,
    alpha: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let mut q: Vec<_> = g.iter().map(|v| -v).collect();
    let mut constant = 0.;
    for (slot, (&j, &a)) in c.positive.iter().zip(alpha).enumerate() {
        let weight = c.nominal_coefficient(p, slot)?;
        constant = finite(constant + weight * a * p.b1[j])?;
        for i in 0..p.input_dim {
            q[i] = finite(q[i] + weight * a * p.w_row(j)[i])?;
        }
    }
    for (bit, &j) in c.negative.iter().enumerate() {
        if mask & (1 << bit) == 0 {
            continue;
        }
        constant = finite(constant + p.c[j] * p.b1[j])?;
        for i in 0..p.input_dim {
            q[i] = finite(q[i] + p.c[j] * p.w_row(j)[i])?;
        }
    }
    Some((q, constant))
}
#[allow(clippy::too_many_arguments)]
pub(super) fn legacy_choose(
    p: &HullMarginProblem,
    residual: &[f64],
    direction: &[f64],
    base_constant: f64,
    bias: f64,
    previous: f64,
    q: &[f64],
    constant: f64,
    proposals: &mut usize,
) -> Option<f64> {
    let mut choices = vec![previous, 0., 1.];
    for i in 0..p.input_dim {
        if direction[i] != 0. {
            let a = -residual[i] / direction[i];
            if a.is_finite() && (0. ..=1.).contains(&a) {
                choices.push(a);
            }
        }
    }
    let mut chosen = previous;
    let mut best = score(p, q, constant)?;
    for a in choices {
        let test: Vec<_> = residual
            .iter()
            .zip(direction)
            .map(|(v, w)| finite(v + a * w))
            .collect::<Option<_>>()?;
        let value = score(p, &test, finite(base_constant + a * bias)?)?;
        *proposals = proposals.checked_add(1)?;
        if value > best {
            best = value;
            chosen = a;
        }
    }
    Some(chosen)
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Work {
    pub masks: usize,
    pub setup: usize,
    pub per_coordinate: usize,
    pub positives: usize,
}
impl Work {
    pub(super) fn cost(self, active: usize) -> Option<usize> {
        if active == 0 || active > self.positives {
            return None;
        }
        self.setup
            .checked_add(active.checked_mul(self.per_coordinate)?)
    }
}
pub(super) fn work(p: &HullMarginProblem, c: &Component) -> Option<Work> {
    if c.negative.is_empty()
        || c.negative.len() > 8
        || c.positive.is_empty()
        || c.positive.len() > MAX_POSITIVE
        || c.positive_units.len() != c.positive.len()
        || c.positive_units.iter().any(|u| !matches!(u, 1 | 2))
    {
        return None;
    }
    let masks = 1usize.checked_shl(c.negative.len() as u32)?;
    Some(Work {
        masks,
        setup: masks
            .checked_mul(c.positive.len() + c.negative.len() + 1)?
            .checked_mul(p.input_dim + 1)?
            .checked_mul(4)?,
        per_coordinate: masks
            .checked_mul(2)?
            .checked_mul(coordinate_work(p.input_dim)?)?,
        positives: c.positive.len(),
    })
}
// Retained single-component API: identical unrestricted coordinate allowance.
pub(super) fn optimize(
    cuts: &HullRootMipCuts<'_>,
    c: &Component,
    g: Vec<f64>,
    b: &Budget,
    next_check_cost: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Component> {
    if !b.can_check() {
        b.stop("checks");
        return None;
    }
    optimize_inner(cuts, c, Some(g), b, next_check_cost, usize::MAX, poll)
}
/// Batch caller has already charged its one whole-plan check and search slot.
/// This remains finite advice; neither partial masks nor its score certify rows.
pub(super) fn optimize_prepaid(
    cuts: &HullRootMipCuts<'_>,
    c: &Component,
    b: &Budget,
    allowance: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Component> {
    optimize_inner(cuts, c, None, b, 0, allowance, poll)
}
fn optimize_inner(
    cuts: &HullRootMipCuts<'_>,
    c: &Component,
    g: Option<Vec<f64>>,
    b: &Budget,
    next_check_cost: usize,
    allowance: usize,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Component> {
    shape(cuts)?;
    let p = cuts.problem;
    let proposed_g = g.as_deref().unwrap_or(&c.g);
    if c.negative.is_empty()
        || c.negative.len() > 8
        || c.positive.len() > MAX_POSITIVE
        || c.positive_units.len() != c.positive.len()
        || c.positive_units.iter().any(|u| !matches!(u, 1 | 2))
        || c.alpha.len()
            != (1usize.checked_shl(c.negative.len() as u32)?).checked_mul(c.positive.len())?
        || c.alpha
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
        || c.negative
            .iter()
            .any(|j| p.c.get(*j).is_none_or(|v| *v >= 0.))
        || c.positive
            .iter()
            .any(|j| p.c.get(*j).is_none_or(|v| *v <= 0.))
        || proposed_g.len() != p.input_dim
        || proposed_g.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    if c.positive.is_empty() {
        b.stop("subset");
        return None;
    }
    let quote = work(cuts.problem, c)?;
    let masks = quote.masks;
    let setup = quote.setup;
    let per_coordinate = quote.per_coordinate;
    let Some(room) = b
        .remaining()?
        .checked_sub(next_check_cost)
        .map(|v| v.min(allowance))
        .and_then(|v| v.checked_sub(setup))
    else {
        b.stop("subset");
        return None;
    };
    let active = c.positive.len().min(room / per_coordinate);
    if active == 0 {
        b.stop("subset");
        return None;
    }
    // Charge the complete bounded attempt before allocating/caching masks.
    // Unoptimized positive roles retain their old finite alpha in EVERY mask.
    b.charge(setup.checked_add(active.checked_mul(per_coordinate)?)?)?;
    let mut trial = c.clone();
    if let Some(g) = g {
        trial.g = g;
    }
    for mask in 0..masks {
        if !live(b.search_end) || !poll() {
            b.stop("deadline");
            return None;
        }
        let alpha = &mut trial.alpha[mask * c.positive.len()..(mask + 1) * c.positive.len()];
        let (mut q, mut constant) = mask_cache(p, c, &trial.g, mask, alpha)?;
        for _ in 0..2 {
            for (slot, &j) in c.positive.iter().enumerate().take(active) {
                if !live(b.search_end) || !poll() {
                    b.stop("deadline");
                    return None;
                }
                // Both proposal paths are precharged as explicit weighted units.
                let previous = alpha[slot];
                let weight = c.nominal_coefficient(p, slot)?;
                let cb = finite(weight * p.b1[j])?;
                let base_constant = finite(constant - previous * cb)?;
                let direction: Vec<_> = p
                    .w_row(j)
                    .iter()
                    .map(|w| finite(weight * w))
                    .collect::<Option<_>>()?;
                let residual: Vec<_> = q
                    .iter()
                    .zip(&direction)
                    .map(|(v, w)| finite(v - previous * w))
                    .collect::<Option<_>>()?;
                let mut proposals = 0;
                let result = if alpha::work(p.input_dim)? < legacy_work(p.input_dim)? {
                    alpha::choose(
                        p,
                        &residual,
                        &direction,
                        base_constant,
                        cb,
                        previous,
                        &q,
                        constant,
                        &mut proposals,
                        b.search_end,
                        poll,
                    )
                } else {
                    legacy_choose(
                        p,
                        &residual,
                        &direction,
                        base_constant,
                        cb,
                        previous,
                        &q,
                        constant,
                        &mut proposals,
                    )
                };
                let mut counts = b.state.lock().ok()?;
                counts.coordinate_proposals = counts.coordinate_proposals.checked_add(proposals)?;
                drop(counts);
                let chosen = result?;
                alpha[slot] = chosen;
                constant = finite(base_constant + chosen * cb)?;
                for i in 0..p.input_dim {
                    q[i] = finite(residual[i] + chosen * direction[i])?;
                }
            }
        }
        if !live(b.search_end) || !poll() {
            b.stop("deadline");
            return None;
        }
        b.state.lock().ok()?.completed_masks += 1;
    }
    Some(trial)
}
