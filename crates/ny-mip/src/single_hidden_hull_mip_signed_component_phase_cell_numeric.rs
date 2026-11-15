// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Nominal pair advice, followed by complete directed closed-cell checking.
use super::*;
use certificate::I;

#[derive(Clone, Copy)]
pub(super) struct Pair {
    pub(super) slot: usize,
    pub(super) ratio: f64,
    pub(super) mass: f64,
    pub(super) score: f64,
}
fn pair(p: &HullMarginProblem, c: &Component, slot: usize, negative: usize) -> Option<(Pair, f64)> {
    let positive = *c.positive.get(slot)?;
    let (mut dot, mut pn, mut nn) = (0., 0., 0.);
    for (&a, &b) in p.w_row(positive).iter().zip(p.w_row(negative)) {
        dot = finite(dot + finite(a * b)?)?;
        pn = finite(pn + finite(a * a)?)?;
        nn = finite(nn + finite(b * b)?)?;
    }
    if pn <= 0. || nn <= 0. {
        return None;
    }
    let ratio = finite(dot / nn)?;
    let quality = finite(finite(dot / pn)? * ratio)?;
    let mass = c.nominal_coefficient(p, slot)?;
    let mut residual = finite(p.b1[positive] - finite(ratio * p.b1[negative])?)?;
    for i in 0..p.input_dim {
        let midpoint = finite(0.5 * p.lower[i] + 0.5 * p.upper[i])?;
        let direction = finite(p.w_row(positive)[i] - finite(ratio * p.w_row(negative)[i])?)?;
        residual = finite(residual + finite(direction * midpoint)?)?;
    }
    let score = finite(finite(mass * residual)? * quality)?;
    (ratio != 0.).then_some((
        Pair {
            slot,
            ratio,
            mass,
            score,
        },
        residual,
    ))
}
pub(super) fn pairs(
    p: &HullMarginProblem,
    c: &Component,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<[[Option<Pair>; 8]; 2]> {
    let mut out = [[None; 8]; 2];
    for (bit, &j) in c.negative.iter().enumerate() {
        for slot in 0..c.positive.len() {
            if !live(end) || !poll() {
                return None;
            }
            if let Some((candidate, _)) = pair(p, c, slot, j) {
                for (side, phase) in out.iter_mut().enumerate() {
                    let sigma = if side == 0 { -1. } else { 1. };
                    let mut candidate = candidate;
                    candidate.score *= (sigma * candidate.ratio).signum();
                    if candidate.score > 0.
                        && phase[bit]
                            .as_ref()
                            .is_none_or(|old: &Pair| candidate.score > old.score)
                    {
                        phase[bit] = Some(candidate);
                    }
                }
            }
        }
    }
    (live(end) && poll()).then_some(out)
}
pub(super) fn add_head(
    p: &HullMarginProblem,
    head: usize,
    coefficient: I,
    q: &mut [I],
    constant: &mut I,
) -> Option<()> {
    *constant = constant.add(coefficient.scale(p.b1[head])?)?;
    for (i, value) in q.iter_mut().enumerate() {
        *value = value.add(coefficient.scale(p.w_row(head)[i])?)?;
    }
    Some(())
}
pub(super) fn price(p: &HullMarginProblem, q: &[I], constant: I) -> Option<f64> {
    let mut lower = constant.lo;
    for (i, coefficient) in q.iter().enumerate() {
        lower = finite(add_down(
            lower,
            coefficient.minimum(p.lower[i], p.upper[i])?,
        ))?;
    }
    Some(lower)
}
fn inactive(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    q: &mut [I],
    mut constant: I,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    for (bit, &head) in c.negative.iter().enumerate() {
        if !poll() {
            return None;
        }
        if mask & (1 << bit) == 0 {
            // On this closed inactive cell z<=0; beta*z is nonpositive.
            add_head(p, head, I::point(-p.c[head])?, q, &mut constant)?;
        }
    }
    price(p, q, constant)
}
fn joint(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    pairs: &[[Option<Pair>; 8]; 2],
    q: &mut [I],
    mut constant: I,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let alpha = c
        .alpha
        .get(mask * c.positive.len()..(mask + 1) * c.positive.len())?;
    let mut selected = None;
    for bit in 0..c.negative.len() {
        if !poll() {
            return None;
        }
        let Some(candidate) = &pairs[(mask >> bit) & 1][bit] else {
            continue;
        };
        let sigma = if mask & (1 << bit) == 0 { -1. } else { 1. };
        let endpoint = if sigma * candidate.ratio > 0. { 1. } else { 0. };
        let delta = endpoint - alpha[candidate.slot];
        let Some(score) = finite(candidate.score * delta.abs()) else {
            continue;
        };
        let beta = candidate.mass * delta * sigma * candidate.ratio;
        if !beta.is_finite() || beta < 0. || score <= 0. {
            continue;
        }
        if selected
            .as_ref()
            .is_none_or(|(_, _, _, _, old)| score > *old)
        {
            selected = Some((bit, candidate.slot, endpoint, beta, score));
        }
    }
    let (bit, slot, endpoint, beta, _) = selected?;
    let positive = c.positive[slot];
    // Enclose exact endpoint-alpha and exact allocated Source mass. The point
    // beta is independently legal; rounded projection identities are NOT proof.
    let correction = I::point(endpoint)?
        .sub(I::point(alpha[slot])?)?
        .scale(p.c[positive])?
        .scale(c.weight(slot)?)?;
    if !poll() {
        return None;
    }
    add_head(p, positive, correction, q, &mut constant)?;
    let signed_beta = if mask & (1 << bit) == 0 { beta } else { -beta };
    if !poll() {
        return None;
    }
    add_head(p, c.negative[bit], I::point(signed_beta)?, q, &mut constant)?;
    price(p, q, constant)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn strengthen(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    coefficients: &[I],
    constant: I,
    base: f64,
    pairs: &[[Option<Pair>; 8]; 2],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, bool)> {
    if !live(end) || !poll() {
        return None;
    }
    let mut cancelled = false;
    let mut sticky = || {
        if cancelled {
            return false;
        }
        let ok = live(end) && poll();
        cancelled |= !ok;
        ok
    };
    // Exactly two D-vector copies; no zero-fill pass precedes the first copy.
    let mut scratch = coefficients.to_vec();
    if scratch.capacity() > coefficients.len() {
        return Some((base, false));
    }
    let first = inactive(p, c, mask, &mut scratch, constant, &mut sticky)
        .unwrap_or(base)
        .max(base);
    if !sticky() {
        return None;
    }
    scratch.clone_from_slice(coefficients);
    let second = joint(p, c, mask, pairs, &mut scratch, constant, &mut sticky).unwrap_or(base);
    if !sticky() {
        return None;
    }
    Some((first.max(second), second > first))
}
