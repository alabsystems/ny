// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Two disjoint positive-slope updates, checked on the same closed phase cell.
use super::*;
use certificate::I;

#[derive(Debug, Clone, Copy)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Quote {
    pub work: usize,
    pub bytes: usize,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SignedPhaseJointReport {
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

pub(in crate::single_hidden_hull::mip_cuts::signed) fn quote(
    d: usize,
    h: usize,
    k: usize,
    terms: usize,
) -> Option<Quote> {
    if !(1..=32).contains(&d) || !(2..=8).contains(&k) {
        return None;
    }
    let m = 1usize << k;
    // Four directed head corrections, one copy/box price, K*(K-1)/2
    // candidate pairs, and a separate complete support search/repair.
    let each = 192usize
        .checked_mul(d + 1)?
        .checked_add(16usize.checked_mul(k * k)?)?
        .checked_add(64usize.checked_mul(k)?)?
        .checked_add(128)?;
    let support = 32usize
        .checked_mul(k * k)?
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
    // Joint runs only with Ray funded, AFTER its numeric scratch is dropped.
    // Its one D-interval copy and scalar candidate selection fit Ray's scratch;
    // support::choose runs sequentially with the same bounded workspace.
    // Persistent additions are one table + its seal, four term copies, and
    // the two simultaneous heap row headers during aggregate/IR publication.
    Some(Quote {
        work: 4096usize
            .checked_add(m.checked_mul(each)?)?
            .checked_add(support)?,
        bytes: 16usize
            .checked_mul(m)?
            .checked_add(64usize.checked_mul(terms)?)?
            .checked_add(2usize.checked_mul(size_of::<RowSpec>())?)?,
    })
}

#[derive(Clone, Copy)]
struct Move {
    bit: usize,
    slot: usize,
    endpoint: f64,
    beta: f64,
    score: f64,
}
fn proposal(
    bit: usize,
    mask: usize,
    alpha: &[f64],
    pairs: &[[Option<numeric::Pair>; 8]; 2],
) -> Option<Move> {
    let pair = pairs[(mask >> bit) & 1][bit]?;
    let sigma = if mask & (1 << bit) == 0 { -1. } else { 1. };
    let endpoint = if sigma * pair.ratio > 0. { 1. } else { 0. };
    let old = *alpha.get(pair.slot)?;
    if !old.is_finite() || !(0. ..=1.).contains(&old) {
        return None;
    }
    let delta = endpoint - old;
    let beta = finite(pair.mass * delta * sigma * pair.ratio)?;
    let score = finite(pair.score * delta.abs())?;
    (beta >= 0. && score > 0.).then_some(Move {
        bit,
        slot: pair.slot,
        endpoint,
        beta,
        score,
    })
}
fn select(
    c: &Component,
    mask: usize,
    pairs: &[[Option<numeric::Pair>; 8]; 2],
    poll: &mut dyn FnMut() -> bool,
) -> Option<[Move; 2]> {
    let alpha = c
        .alpha
        .get(mask * c.positive.len()..(mask + 1) * c.positive.len())?;
    let mut best: Option<([Move; 2], f64)> = None;
    for i in 0..c.negative.len() {
        if !poll() {
            return None;
        }
        let Some(first) = proposal(i, mask, alpha, pairs) else {
            continue;
        };
        for j in i + 1..c.negative.len() {
            if !poll() {
                return None;
            }
            let Some(second) = proposal(j, mask, alpha, pairs) else {
                continue;
            };
            let Some(score) = finite(first.score + second.score) else {
                continue;
            };
            // Applying two endpoint corrections to the same alpha is illegal.
            if first.slot != second.slot && best.is_none_or(|(_, old)| score > old) {
                best = Some(([first, second], score));
            }
        }
    }
    best.map(|(moves, _)| moves)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn strengthen(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    coefficients: &[I],
    constant: I,
    old: f64,
    pairs: &[[Option<numeric::Pair>; 8]; 2],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let mut cancelled = false;
    let mut sticky = || {
        if cancelled {
            return false;
        }
        let ok = live(end) && poll();
        cancelled |= !ok;
        ok
    };
    let candidate = (|| {
        if !sticky() || coefficients.len() != p.input_dim || c.negative.len() > 8 {
            return None;
        }
        let moves = select(c, mask, pairs, &mut sticky)?;
        let alpha = c
            .alpha
            .get(mask * c.positive.len()..(mask + 1) * c.positive.len())?;
        let mut q = coefficients.to_vec();
        if q.capacity() > coefficients.len() {
            return None;
        }
        let mut constant = constant;
        for step in moves {
            if !sticky() {
                return None;
            }
            let head = c.positive[step.slot];
            // The original allocated Source coefficient is the authority.
            // Nominal pair mass/ratio merely proposed a finite nonnegative beta.
            let delta = I::point(step.endpoint)?
                .sub(I::point(alpha[step.slot])?)?
                .scale(p.c[head])?
                .scale(c.weight(step.slot)?)?;
            numeric::add_head(p, head, delta, &mut q, &mut constant)?;
            if !sticky() {
                return None;
            }
            let signed = if mask & (1 << step.bit) == 0 {
                step.beta
            } else {
                -step.beta
            };
            numeric::add_head(
                p,
                c.negative[step.bit],
                I::point(signed)?,
                &mut q,
                &mut constant,
            )?;
        }
        numeric::price(p, &q, constant)
    })()
    .unwrap_or(old);
    sticky().then_some(old.max(candidate))
}
