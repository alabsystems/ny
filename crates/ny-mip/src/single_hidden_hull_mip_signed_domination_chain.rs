// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Nominal Lovasz-chain direction for one learned mixed-head residual.
//!
//! This returns ONLY a point slope. It does not return or certify an intercept,
//! a row, primal feasibility, or a verdict. The original-Source support kernel
//! must independently compute H for the original W/b and lambda. The caller
//! must pay construction, ordering, residual fitting and simultaneous payload
//! before this optional advice; the caller also owns the original clock/poll.
const MAX_INPUTS: usize = 128;
#[derive(Clone, Copy)]
struct Entry {
    input: usize,
    width: f64,
    increment: f64,
    normalized: f64,
    sign: f64,
}
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// Same exact comparator as the original insertion helper; only indices move.
/// At most ceil(log2(n)) levels, each with <=n comparisons/reads/writes.
/// Both index buffers are retained in the prepaid simultaneous payload.
fn merge_order(entries: &[Entry]) -> Vec<usize> {
    let n = entries.len();
    let mut order: Vec<_> = (0..n).collect();
    let mut scratch = vec![0; n];
    let mut width = 1;
    while width < n {
        let mut start = 0;
        while start < n {
            let middle = (start + width).min(n);
            let end = (middle + width).min(n);
            let (mut left, mut right) = (start, middle);
            for slot in &mut scratch[start..end] {
                let take_left = if right == end {
                    true
                } else if left == middle {
                    false
                } else {
                    let a = entries[order[left]];
                    let b = entries[order[right]];
                    a.normalized > b.normalized
                        || (a.normalized == b.normalized && a.input <= b.input)
                };
                *slot = if take_left {
                    let i = order[left];
                    left += 1;
                    i
                } else {
                    let i = order[right];
                    right += 1;
                    i
                };
            }
            start = end;
        }
        std::mem::swap(&mut order, &mut scratch);
        width *= 2;
    }
    order
}

/// Select a nominal chain slope; clamping affects advice order only.
/// Finite-arithmetic loss may weaken it: no computed slope has proof authority.
pub(super) fn nominal_chain_direction(
    c: &[f64],
    d: f64,
    lower: &[f64],
    upper: &[f64],
    advice: &[f64],
) -> Option<Vec<f64>> {
    let n = c.len();
    if n > MAX_INPUTS || lower.len() != n || upper.len() != n || advice.len() != n || !d.is_finite()
    {
        return None;
    }
    for i in 0..n {
        if !c[i].is_finite()
            || !lower[i].is_finite()
            || !upper[i].is_finite()
            || !advice[i].is_finite()
            || lower[i] > upper[i]
        {
            return None;
        }
    }
    let mut value = d;
    let mut order: Vec<Entry> = Vec::with_capacity(n);
    for i in 0..n {
        if c[i] == 0. {
            continue;
        }
        let (base, sign) = if c[i] > 0. {
            (lower[i], 1.)
        } else {
            (upper[i], -1.)
        };
        value = finite(value + finite(c[i] * base)?)?;
        if lower[i] == upper[i] {
            continue;
        }
        let width = finite(upper[i] - lower[i])?;
        let increment = finite(c[i].abs() * width)?;
        // A positive real increment lost to underflow is declined, not used as
        // an exact zero or as authority. The caller retains its old direction.
        if width <= 0. || increment <= 0. {
            return None;
        }
        let point = advice[i].clamp(lower[i], upper[i]);
        let offset = finite(if sign > 0. {
            point - lower[i]
        } else {
            upper[i] - point
        })?;
        let normalized = finite(offset / width)?.clamp(0., 1.);
        let entry = Entry {
            input: i,
            width,
            increment,
            normalized,
            sign,
        };
        // Append preserves every old normalized bit and original-index tie.
        order.push(entry);
    }
    let indices = merge_order(&order);

    let mut slope = vec![0.; n];
    let mut previous = value.max(0.);
    for index in indices {
        let entry = order[index];
        value = finite(value + entry.increment)?;
        let next = value.max(0.);
        let delta = finite(next - previous)?;
        if delta < 0. {
            return None;
        }
        slope[entry.input] = finite(entry.sign * finite(delta / entry.width)?)?;
        previous = next;
    }
    Some(slope)
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_chain_tests.rs"]
mod tests;
