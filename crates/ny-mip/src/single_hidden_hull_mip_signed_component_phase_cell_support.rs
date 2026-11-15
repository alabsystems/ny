// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded coordinate advice for the complete-table support dual.
use super::*;

pub(super) fn optimize(
    values: &[f64],
    mut coefficients: [f64; 8],
    mapped: &[bool],
    poll: &mut dyn FnMut() -> bool,
) -> Option<[f64; 8]> {
    let k = mapped.len();
    if !(1..=8).contains(&k) || values.len() != 1 << k {
        return None;
    }
    for _ in 0..2 {
        for j in 0..k {
            if !poll() {
                return None;
            }
            if !mapped[j] {
                coefficients[j] = 0.;
                continue;
            }
            let mut sides = [f64::INFINITY; 2];
            for (mask, value) in values.iter().enumerate() {
                if !poll() {
                    return None;
                }
                let mut residual = finite(2. * value)?;
                for (i, a) in coefficients[..k].iter().enumerate() {
                    if i != j && mask & (1 << i) != 0 {
                        residual = finite(residual + a)?;
                    }
                }
                let side = (mask >> j) & 1;
                sides[side] = sides[side].min(residual);
            }
            // Equalize the two affine branches of the one-coordinate minimum.
            coefficients[j] = finite(sides[0] - sides[1])?;
        }
    }
    Some(coefficients)
}

pub(super) fn score(lower: f64, a: &[f64], activity: Option<&[f64]>) -> Option<certificate::I> {
    let mut value = certificate::I::point(lower)?;
    for (i, coefficient) in a.iter().enumerate() {
        let s = activity.and_then(|v| v.get(i)).copied().unwrap_or(0.);
        value = value.sub(certificate::I::point(*coefficient)?.scale(s)?)?;
    }
    Some(value)
}

pub(super) fn choose(
    values: &[f64],
    coefficients: [f64; 8],
    lower: f64,
    mapped: &[bool],
    activity: Option<&[f64]>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<([f64; 8], f64)> {
    let mut cancelled = false;
    let mut sticky = || {
        if cancelled {
            return false;
        }
        let ok = live(end) && poll();
        cancelled |= !ok;
        ok
    };
    let k = mapped.len();
    let better = (|| {
        let proposal = optimize(values, coefficients, mapped, &mut sticky)?;
        // Nominal equalization supplies no authority: repair and separately
        // verify the actual serialized coefficients against every mask.
        let next = repair(values, &proposal[..k], end, &mut sticky)?;
        (score(next, &proposal[..k], activity)?.lo > score(lower, &coefficients[..k], activity)?.hi)
            .then_some((proposal, next))
    })();
    sticky().then_some(better.unwrap_or((coefficients, lower)))
}
