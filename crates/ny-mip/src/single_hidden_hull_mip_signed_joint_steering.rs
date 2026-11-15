// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Finite nominal proposals only; every joint mask is independently replayed.
use super::*;
fn score(p: &HullMarginProblem, c: &Component, slot: usize) -> Option<f64> {
    let j = *c.positive.get(slot)?;
    let mass = c.nominal_coefficient(p, slot)?;
    let mut value = finite(mass * p.b1[j].abs())?;
    for i in 0..p.input_dim {
        value = finite(value + mass * p.w_row(j)[i].abs() * (p.upper[i] - p.lower[i]))?;
    }
    Some(value)
}
fn selected(p: &HullMarginProblem, c: &Component) -> Option<Vec<usize>> {
    let mut result = Vec::with_capacity(4);
    for _ in 0..c.positive.len().min(4) {
        let mut best = None;
        for slot in 0..c.positive.len() {
            if result.contains(&slot) {
                continue;
            }
            let value = score(p, c, slot)?;
            if best.is_none_or(|(_, old)| value > old) {
                best = Some((slot, value));
            }
        }
        if let Some((slot, _)) = best {
            result.push(slot);
        }
    }
    Some(result)
}
fn value(
    p: &HullMarginProblem,
    constant: f64,
    residual: &[f64],
    q: &[f64],
    qb: f64,
    t: f64,
) -> Option<f64> {
    let mut v = finite(constant + qb * t)?;
    for i in 0..p.input_dim {
        let a = finite(residual[i] + q[i] * t)?;
        v = finite(
            v + if a >= 0. {
                a * p.lower[i]
            } else {
                a * p.upper[i]
            },
        )?;
    }
    Some(v)
}
/// <=4 greatest-weight zero crossings plus endpoints/current/midpoint. This
/// is bounded generic steering, not an exact LP claim or trusted optimizer.
fn coordinate(
    p: &HullMarginProblem,
    c: &Component,
    slot: usize,
    constant: &mut f64,
    residual: &mut [f64],
    alpha: &mut [f64],
) -> Option<()> {
    let j = c.positive[slot];
    let mass = c.nominal_coefficient(p, slot)?;
    let qb = finite(mass * p.b1[j])?;
    let q: Vec<_> = p
        .w_row(j)
        .iter()
        .map(|w| finite(mass * w))
        .collect::<Option<_>>()?;
    let old = alpha[slot];
    let base = finite(*constant - qb * old)?;
    let r: Vec<_> = residual
        .iter()
        .zip(&q)
        .map(|(v, q)| finite(v - q * old))
        .collect::<Option<_>>()?;
    let mut crossings: [Option<(f64, f64)>; 4] = [None; 4];
    for i in 0..p.input_dim {
        if q[i] == 0. {
            continue;
        }
        let point = finite(-r[i] / q[i])?;
        if !(0. ..=1.).contains(&point) {
            continue;
        }
        let weight = finite(q[i].abs() * (p.upper[i] - p.lower[i]))?;
        let mut carry = Some((point, weight));
        for cell in &mut crossings {
            let Some((_, weight)) = carry else {
                break;
            };
            if cell.is_none_or(|(_, old)| weight > old) {
                std::mem::swap(cell, &mut carry);
            }
        }
    }
    let mut best = (old, value(p, base, &r, &q, qb, old)?);
    for candidate in [Some(0.), Some(1.), Some(0.5)]
        .into_iter()
        .chain(crossings.into_iter().map(|entry| entry.map(|v| v.0)))
    {
        let Some(candidate) = candidate else {
            continue;
        };
        let v = value(p, base, &r, &q, qb, candidate)?;
        if v > best.1 {
            best = (candidate, v);
        }
    }
    alpha[slot] = best.0;
    *constant = finite(base + qb * best.0)?;
    for i in 0..p.input_dim {
        residual[i] = finite(r[i] + q[i] * best.0)?;
    }
    Some(())
}
pub(super) fn steer(
    p: &HullMarginProblem,
    c: &mut Component,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
    coordinates: &mut usize,
) -> Option<()> {
    steer_with_tail(p, c, None, end, poll, coordinates)
}
/// Same nominal arithmetic; a partial union includes every omitted affine head.
pub(super) fn steer_with_tail(
    p: &HullMarginProblem,
    c: &mut Component,
    tail: Option<(f64, &[f64])>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
    coordinates: &mut usize,
) -> Option<()> {
    let slots = selected(p, c)?;
    let masks = 1usize.checked_shl(c.negative.len() as u32)?;
    let positives = c.positive.len();
    for mask in 0..masks {
        if !live(end) || !poll() {
            return None;
        }
        let mut alpha = c.alpha[mask * positives..(mask + 1) * positives].to_vec();
        let mut residual: Vec<_> = c.g.iter().map(|v| -v).collect();
        let mut constant = tail.map_or(0., |v| v.0);
        if let Some((_, coeff)) = tail {
            if coeff.len() != p.input_dim {
                return None;
            }
            for (v, a) in residual.iter_mut().zip(coeff) {
                *v = finite(*v + a)?;
            }
        }
        for (slot, &j) in c.positive.iter().enumerate() {
            let mass = finite(c.nominal_coefficient(p, slot)? * alpha[slot])?;
            constant = finite(constant + mass * p.b1[j])?;
            for i in 0..p.input_dim {
                residual[i] = finite(residual[i] + mass * p.w_row(j)[i])?;
            }
        }
        for (bit, &j) in c.negative.iter().enumerate() {
            if mask & (1 << bit) == 0 {
                continue;
            }
            constant = finite(constant + p.c[j] * p.b1[j])?;
            for i in 0..p.input_dim {
                residual[i] = finite(residual[i] + p.c[j] * p.w_row(j)[i])?;
            }
        }
        for &slot in &slots {
            if !live(end) || !poll() {
                return None;
            }
            coordinate(p, c, slot, &mut constant, &mut residual, &mut alpha)?;
            *coordinates = coordinates.checked_add(1)?;
        }
        if alpha
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
        {
            return None;
        }
        c.alpha[mask * positives..(mask + 1) * positives].copy_from_slice(&alpha);
    }
    (live(end) && poll()).then_some(())
}
