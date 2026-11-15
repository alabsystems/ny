// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Sorted nominal alpha breakpoints. Every result remains untrusted advice.
use super::*;

#[derive(Clone, Copy, Debug)]
struct Event {
    at: f64,
    drop: f64,
}

pub(in super::super) fn work(d: usize) -> Option<usize> {
    let levels = usize::BITS as usize - (d.max(2) - 1).leading_zeros() as usize;
    72usize
        .checked_mul(d.checked_add(1)?)?
        .checked_add(16usize.checked_mul(d)?.checked_mul(levels)?)
}

fn alive(end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    (live(end) && poll()).then_some(())
}

fn greater(a: Event, b: Event) -> bool {
    a.at.total_cmp(&b.at)
        .then_with(|| a.drop.total_cmp(&b.drop))
        .is_gt()
}

fn sift(events: &mut [Event], mut root: usize, limit: usize) {
    while root < limit / 2 {
        let mut child = 2 * root + 1;
        if child + 1 < limit && greater(events[child + 1], events[child]) {
            child += 1;
        }
        if !greater(events[child], events[root]) {
            return;
        }
        events.swap(root, child);
        root = child;
    }
}

/// In-place bounded heap sort: no library scratch or uncharged D-vector scores.
fn sort(events: &mut [Event], end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    for root in (0..events.len() / 2).rev() {
        alive(end, poll)?;
        let limit = events.len();
        sift(events, root, limit);
    }
    for limit in (1..events.len()).rev() {
        alive(end, poll)?;
        events.swap(0, limit);
        sift(events, 0, limit);
    }
    Some(())
}

fn plateau(events: &[Event], mut slope: f64) -> Option<(f64, f64)> {
    if slope < 0. {
        return Some((0., 0.));
    }
    if slope == 0. {
        return Some((0., events.first().map_or(1., |e| e.at)));
    }
    let mut i = 0;
    while i < events.len() {
        let point = events[i].at;
        let mut drop = 0.;
        while i < events.len() && events[i].at == point {
            drop = finite(drop + events[i].drop)?;
            i += 1;
        }
        slope = finite(slope - drop)?;
        if slope < 0. {
            return Some((point, point));
        }
        if slope == 0. {
            return Some((point, events.get(i).map_or(1., |e| e.at)));
        }
    }
    Some((1., 1.))
}

/// At most FIVE complete direct scores, including the unchanged cache reference.
/// Equal/worse advice preserves the old alpha bits. Ratios/slopes are not proof.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn choose(
    p: &HullMarginProblem,
    residual: &[f64],
    direction: &[f64],
    base_constant: f64,
    bias: f64,
    previous: f64,
    q: &[f64],
    constant: f64,
    proposals: &mut usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let d = p.input_dim;
    if d == 0
        || d > 128
        || [
            residual.len(),
            direction.len(),
            q.len(),
            p.lower.len(),
            p.upper.len(),
        ]
        .into_iter()
        .any(|n| n != d)
        || !previous.is_finite()
        || !(0. ..=1.).contains(&previous)
        || [base_constant, bias, constant]
            .into_iter()
            .any(|v| !v.is_finite())
    {
        return None;
    }
    let mut events = Vec::with_capacity(d);
    let mut slope = bias;
    for i in 0..d {
        alive(end, poll)?;
        let (r, v, l, u) = (
            finite(residual[i])?,
            finite(direction[i])?,
            finite(p.lower[i])?,
            finite(p.upper[i])?,
        );
        if l > u {
            return None;
        }
        let width = finite(u - l)?;
        let endpoint = if r > 0. || (r == 0. && v >= 0.) { l } else { u };
        slope = finite(slope + finite(v * endpoint)?)?;
        if v == 0. || width == 0. {
            continue;
        }
        let at = finite(-r / v)?;
        if 0. < at && at < 1. {
            events.push(Event {
                at,
                drop: finite(v.abs() * width)?,
            });
        }
    }
    sort(&mut events, end, poll)?;
    let (left, right) = plateau(&events, slope)?;
    let mut chosen = previous;
    let mut best = score(p, q, constant)?;
    let mut seen = [previous, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    let mut used = 1;
    for a in [0., 1., left, right] {
        alive(end, poll)?;
        if !a.is_finite() || !(0. ..=1.).contains(&a) {
            return None;
        }
        if seen[..used].contains(&a) {
            continue;
        }
        seen[used] = a;
        used += 1;
        let test = residual
            .iter()
            .zip(direction)
            .map(|(r, v)| finite(r + a * v))
            .collect::<Option<Vec<_>>>()?;
        let value = score(p, &test, finite(base_constant + a * bias)?)?;
        *proposals = proposals.checked_add(1)?;
        if value > best {
            best = value;
            chosen = a;
        }
    }
    alive(end, poll)?;
    Some(chosen)
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_alpha_tests.rs"]
mod tests;
