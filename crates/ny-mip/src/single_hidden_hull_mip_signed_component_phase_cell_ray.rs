// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Finite one-coordinate sign-multiplier advice, independently repriced.
use super::*;
use certificate::I;

#[derive(Debug, Clone, Copy)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Quote {
    pub work: usize,
    pub bytes: usize,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SignedPhaseRayReport {
    pub admitted: bool,
    pub work_units: usize,
    pub admitted_bytes: usize,
    pub masks: usize,
    pub improved_masks: usize,
    pub prepared_rows: usize,
    pub prepared_terms: usize,
    pub prepared_selectors: usize,
    pub appended_rows: usize,
}
/// Pure bounded quote arithmetic. No Source or table walk precedes payment.
pub(in crate::single_hidden_hull::mip_cuts::signed) fn quote(
    d: usize,
    h: usize,
    k: usize,
    terms: usize,
) -> Option<Quote> {
    if !(1..=32).contains(&d) || !(1..=8).contains(&k) {
        return None;
    }
    let m = 1usize << k;
    let each = 16usize
        .checked_mul(k)?
        .checked_mul(d + 1)?
        .checked_add(4usize.checked_mul(d)?.checked_mul(d - 1)?)?
        .checked_add(128usize.checked_mul(d + 1)?)?
        .checked_add(128)?;
    let support = 32usize
        .checked_mul(k)?
        .checked_mul(k)?
        .checked_mul(m)?
        .checked_add(128usize.checked_mul(k)?.checked_mul(m)?)?
        .checked_add(64usize.checked_mul(m)?)?
        .checked_add(128usize.checked_mul((k + 1).checked_pow(2)?)?)?
        .checked_add(
            256usize.checked_mul(
                terms
                    .checked_add(2 * k)?
                    .checked_add(h)?
                    .checked_add(d)?
                    .checked_add(2)?,
            )?,
        )?
        .checked_add(4096)?;
    // One f64 table and its u64 integrity copy, moved rather than cloned.
    let bytes = 16usize
        .checked_mul(m)?
        .checked_add(64usize.checked_mul(terms)?)?
        .checked_add(128usize.checked_mul(k)?)?
        .checked_add(2048)?
        .checked_add(64usize.checked_mul(d + 1)?)?
        .checked_add(4096)?;
    Some(Quote {
        work: 4096usize
            .checked_add(m.checked_mul(each)?)?
            .checked_add(support)?,
        bytes,
    })
}

fn midpoint(x: I) -> Option<f64> {
    finite(0.5 * x.lo + 0.5 * x.hi)
}
fn slope(
    p: &HullMarginProblem,
    head: usize,
    sigma: f64,
    q: &[I],
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let mut slope = finite(-sigma * p.b1[head])?;
    for (i, a) in q.iter().enumerate() {
        if !poll() {
            return None;
        }
        let a = midpoint(*a)?;
        let direction = -sigma * p.w_row(head)[i];
        // At a zero coefficient take the right-hand derivative of the minimum.
        let x = if a > 0. || (a == 0. && direction >= 0.) {
            p.lower[i]
        } else {
            p.upper[i]
        };
        slope = finite(slope + finite(direction * x)?)?;
    }
    Some(slope)
}
pub(super) fn proposal(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    q: &[I],
    poll: &mut dyn FnMut() -> bool,
) -> Option<(usize, f64)> {
    if q.len() != p.input_dim || !(1..=32).contains(&q.len()) {
        return None;
    }
    let mut selected = None;
    for (bit, &head) in c.negative.iter().enumerate() {
        if !poll() {
            return None;
        }
        let sigma = if mask & (1 << bit) == 0 { -1. } else { 1. };
        let derivative = slope(p, head, sigma, q, poll)?;
        let score = finite(derivative * -p.c[head])?;
        if derivative > 0. && selected.is_none_or(|(_, _, _, old)| score > old) {
            selected = Some((head, sigma, derivative, score));
        }
    }
    let (head, sigma, mut derivative, _) = selected?;
    let mut points = [(0., 0.); 32];
    let mut len = 0;
    for (i, a) in q.iter().enumerate() {
        if !poll() {
            return None;
        }
        let w = sigma * p.w_row(head)[i];
        if w == 0. {
            continue;
        }
        let beta = finite(midpoint(*a)? / w)?;
        if beta <= 0. {
            continue;
        }
        let jump = finite(w.abs() * finite(p.upper[i] - p.lower[i])?)?;
        points[len] = (beta, jump);
        let mut j = len;
        while j > 0 && points[j].0 < points[j - 1].0 {
            if !poll() {
                return None;
            }
            points.swap(j, j - 1);
            j -= 1;
        }
        len += 1;
    }
    // A positive unbounded nominal tail is not a certificate of infeasibility.
    // Choose a finite legal point; the directed reprice below is the authority.
    let mut beta = finite(-p.c[head])?;
    for &(point, jump) in &points[..len] {
        if !poll() {
            return None;
        }
        beta = point;
        derivative = finite(derivative - jump)?;
        if derivative <= 0. {
            break;
        }
    }
    (beta >= 0.).then_some((head, -sigma * beta))
}

#[cfg(test)]
pub(super) fn strengthen(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    coefficients: &[I],
    constant: I,
    old: f64,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    strengthen_with_advice(p, c, mask, coefficients, constant, old, end, poll)
        .map(|(value, _)| value)
}

/// Return the same checked floor and the already-paid nominal proposal.
/// A consumer must independently validate and reprice this advice for new g.
#[allow(clippy::too_many_arguments)]
pub(super) fn strengthen_with_advice(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    coefficients: &[I],
    constant: I,
    old: f64,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, Option<(usize, f64)>)> {
    let mut cancelled = false;
    let mut sticky = || {
        if cancelled {
            return false;
        }
        let ok = live(end) && poll();
        cancelled |= !ok;
        ok
    };
    let mut advice = None;
    let candidate = (|| {
        if !sticky() {
            return None;
        }
        let (head, beta) = proposal(p, c, mask, coefficients, &mut sticky)?;
        advice = Some((head, beta));
        let mut q = coefficients.to_vec();
        if q.capacity() > coefficients.len() {
            return None;
        }
        let mut constant = constant;
        if !sticky() {
            return None;
        }
        numeric::add_head(p, head, I::point(beta)?, &mut q, &mut constant)?;
        numeric::price(p, &q, constant)
    })()
    .unwrap_or(old);
    sticky().then_some((old.max(candidate), advice))
}
