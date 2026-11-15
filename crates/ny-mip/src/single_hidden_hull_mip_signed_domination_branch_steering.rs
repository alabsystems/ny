// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Four nominal one-dimensional supports. Fresh Source owns every final bound.
use std::time::Instant;

pub(super) struct Affines<'a> {
    pub a: [&'a [f64]; 2],
    pub bias: [f64; 2],
    pub b: &'a [f64],
    pub b_bias: f64,
    pub v: &'a [f64],
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub point: &'a [f64],
}
pub(super) struct Minimum {
    pub tau: [f64; 4],
    pub upper: f64,
    pub price: f64,
    pub mask_bounds: [f64; 4],
}
struct Entry {
    at: f64,
    weight: f64,
    coordinate: usize,
}
struct Workspace {
    weights: Vec<f64>,
    slope: f64,
    entries: Vec<Entry>,
    order: Vec<usize>,
    scratch: Vec<usize>,
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn live(end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    (Instant::now() < end && poll()).then_some(())
}
fn validate(a: &Affines<'_>) -> Option<usize> {
    let d = a.b.len();
    (d > 0
        && d <= 128
        && a.a.iter().all(|v| v.len() == d)
        && a.v.len() == d
        && a.lower.len() == d
        && a.upper.len() == d
        && a.point.len() == d
        && a.bias.iter().chain([&a.b_bias]).all(|x| x.is_finite()))
    .then_some(d)
}
impl Workspace {
    fn new(a: &Affines<'_>, end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<Self> {
        let d = validate(a)?;
        let mut weights = Vec::with_capacity(d);
        let mut slope = -a.b_bias;
        for i in 0..d {
            live(end, poll)?;
            if !a.lower[i].is_finite()
                || !a.upper[i].is_finite()
                || a.lower[i] > a.upper[i]
                || !a.b[i].is_finite()
                || !a.v[i].is_finite()
                || !a.point[i].is_finite()
            {
                return None;
            }
            let mid = finite(0.5 * a.lower[i] + 0.5 * a.upper[i])?;
            let radius = finite(0.5 * a.upper[i] - 0.5 * a.lower[i])?;
            weights.push(finite(radius * a.b[i].abs())?);
            // Fixed coordinates still contribute to this full linear slope.
            slope = finite(slope - finite(mid * a.b[i])?)?;
        }
        Some(Self {
            weights,
            slope,
            entries: Vec::with_capacity(d),
            order: Vec::with_capacity(d),
            scratch: Vec::with_capacity(d),
        })
    }
    fn sorted(&mut self, end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
        self.order.clear();
        self.order.extend(0..self.entries.len());
        self.scratch.clear();
        self.scratch.resize(self.entries.len(), 0);
        let mut width = 1;
        while width < self.order.len() {
            let mut start = 0;
            while start < self.order.len() {
                let mid = (start + width).min(self.order.len());
                let stop = (mid + width).min(self.order.len());
                let (mut left, mut right) = (start, mid);
                for out in start..stop {
                    live(end, poll)?;
                    let first = if left == mid {
                        false
                    } else if right == stop {
                        true
                    } else {
                        let a = &self.entries[self.order[left]];
                        let b = &self.entries[self.order[right]];
                        a.at < b.at || (a.at == b.at && a.coordinate <= b.coordinate)
                    };
                    self.scratch[out] = if first {
                        let j = self.order[left];
                        left += 1;
                        j
                    } else {
                        let j = self.order[right];
                        right += 1;
                        j
                    };
                }
                start = stop;
            }
            std::mem::swap(&mut self.order, &mut self.scratch);
            width *= 2;
        }
        Some(())
    }
    fn crossing(
        &self,
        mut slope: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        if slope >= 0. {
            return Some(0.);
        }
        for &j in &self.order {
            live(end, poll)?;
            let e = &self.entries[j];
            if e.at <= 0. {
                continue;
            }
            if e.at > 1. {
                break;
            }
            slope = finite(slope + finite(2. * e.weight)?)?;
            if slope >= 0. {
                return Some(e.at.clamp(0., 1.));
            }
        }
        Some(1.)
    }
}
fn coefficient(a: &Affines<'_>, mask: usize, i: usize) -> Option<f64> {
    let mut c = 0.;
    for t in 0..2 {
        if mask & (1 << t) != 0 {
            c = finite(c + a.a[t][i])?;
        }
    }
    finite(c - a.v[i])
}
fn intercept(a: &Affines<'_>, mask: usize, tau: f64) -> Option<f64> {
    let mut c = 0.;
    for t in 0..2 {
        if mask & (1 << t) != 0 {
            c = finite(c + a.bias[t])?;
        }
    }
    finite(c - finite(tau * a.b_bias)?)
}
fn endpoint(c: f64, l: f64, u: f64) -> Option<f64> {
    finite(c * l).and_then(|a| finite(c * u).map(|b| a.max(b)))
}
fn prepare_mask(
    a: &Affines<'_>,
    w: &mut Workspace,
    mask: usize,
    old: f64,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, f64)> {
    w.entries.clear();
    let mut slope = w.slope;
    let mut old_bound = intercept(a, mask, old)?;
    for i in 0..a.b.len() {
        live(end, poll)?;
        let c = coefficient(a, mask, i)?;
        let old_c = finite(c - finite(old * a.b[i])?)?;
        old_bound = finite(old_bound + endpoint(old_c, a.lower[i], a.upper[i])?)?;
        if a.b[i] == 0. || w.weights[i] == 0. {
            continue;
        }
        let at = finite(c / a.b[i])?;
        let weight = w.weights[i];
        // Include every out-of-range breakpoint in the initial derivative.
        slope = finite(if at <= 0. {
            slope + weight
        } else {
            slope - weight
        })?;
        w.entries.push(Entry {
            at,
            weight,
            coordinate: i,
        });
    }
    Some((slope, old_bound))
}
fn support(
    a: &Affines<'_>,
    mask: usize,
    tau: f64,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let mut h = intercept(a, mask, tau)?;
    for i in 0..a.b.len() {
        live(end, poll)?;
        let c = finite(coefficient(a, mask, i)? - finite(tau * a.b[i])?)?;
        h = finite(h + endpoint(c, a.lower[i], a.upper[i])?)?;
    }
    Some(h)
}
/// One shared workspace, four stable sorts and four fresh nominal reprices.
/// Rounded breakpoints select finite point coefficients, never proof authority.
pub(super) fn minimize(
    a: &Affines<'_>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Minimum> {
    live(end, poll)?;
    let mut w = Workspace::new(a, end, poll)?;
    let mut tau = [0., 0.5, 0.5, 1.];
    let mut mask_bounds = [0.; 4];
    for mask in 0..4 {
        live(end, poll)?;
        let (slope, old) = prepare_mask(a, &mut w, mask, tau[mask], end, poll)?;
        w.sorted(end, poll)?;
        let chosen = w.crossing(slope, end, poll)?;
        let bound = support(a, mask, chosen, end, poll)?;
        if bound < old {
            tau[mask] = chosen;
            mask_bounds[mask] = bound;
        } else {
            mask_bounds[mask] = old;
        }
    }
    let upper = mask_bounds
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut at = 0.;
    for (&v, &x) in a.v.iter().zip(a.point) {
        live(end, poll)?;
        at = finite(at + finite(v * x)?)?;
    }
    live(end, poll)?;
    Some(Minimum {
        tau,
        upper,
        price: finite(upper + at)?,
        mask_bounds,
    })
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_branch_steering_tests.rs"]
mod tests;
