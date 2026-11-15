// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Cached one-pass seed. Only immutable, bit-identical subexpressions are reused.
use super::*;
struct Coordinates {
    midpoint: Vec<f64>,
    halfwidth: Vec<f64>,
    target: Vec<f64>,
    centered: f64,
}
struct Feature {
    head: usize,
    centered: f64,
    denominator: f64,
}
pub(super) struct Seed {
    pub positive: Vec<usize>,
    pub lambda: Vec<f64>,
    pub residual: Vec<f64>,
    pub bias: f64,
}
impl Coordinates {
    fn new(
        p: &HullMarginProblem,
        head: usize,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        let d = p.input_dim;
        let (mut midpoint, mut halfwidth, mut target) = (
            Vec::with_capacity(d),
            Vec::with_capacity(d),
            Vec::with_capacity(d),
        );
        let mut centered = p.b1[head];
        for i in 0..d {
            if !live(end) || !poll() {
                return None;
            }
            let mid = center(p.lower[i], p.upper[i])?;
            let width = finite(0.5 * p.upper[i] - 0.5 * p.lower[i])?;
            centered = finite(centered + p.w_row(head)[i] * mid)?;
            midpoint.push(mid);
            halfwidth.push(width);
            target.push(finite(p.w_row(head)[i] * width)?);
        }
        Some(Self {
            midpoint,
            halfwidth,
            target,
            centered,
        })
    }
    fn feature(
        &self,
        p: &HullMarginProblem,
        point: &Point,
        j: usize,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<(Feature, f64)> {
        let mut pb = p.b1[j];
        let (mut num, mut den) = (0., 0.);
        for i in 0..p.input_dim {
            if !live(end) || !poll() {
                return None;
            }
            pb = finite(pb + p.w_row(j)[i] * self.midpoint[i])?;
            let b = finite(p.w_row(j)[i] * self.halfwidth[i])?;
            num = finite(num + finite(self.target[i] * b)?)?;
            den = finite(den + finite(b * b)?)?;
        }
        num = finite(num + finite(self.centered * pb)?)?;
        den = finite(den + finite(pb * pb)?)?;
        let score = if num <= 0. || den <= 0. {
            0.
        } else {
            let estimate = finite(num / den)?.clamp(0., 256.);
            let penalty = finite(1. + finite(estimate * finite(point.values[j])?.max(0.))?)?;
            finite(finite(estimate * num)? / penalty)?
        };
        Some((
            Feature {
                head: j,
                centered: pb,
                denominator: den,
            },
            score,
        ))
    }
    fn fit(
        &self,
        p: &HullMarginProblem,
        feature: &Feature,
        seed: &mut Seed,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let j = feature.head;
        let (mut rb, mut num) = (seed.bias, 0.);
        for i in 0..p.input_dim {
            if !live(end) || !poll() {
                return None;
            }
            rb = finite(rb + seed.residual[i] * self.midpoint[i])?;
            let a = finite(seed.residual[i] * self.halfwidth[i])?;
            let b = finite(p.w_row(j)[i] * self.halfwidth[i])?;
            num = finite(num + finite(a * b)?)?;
        }
        num = finite(num + finite(rb * feature.centered)?)?;
        let gain = if feature.denominator > 0. {
            finite(num / feature.denominator)?.clamp(0., 256.)
        } else {
            0.
        };
        // Retain zero partners for the projected step, but DO NOT subtract
        // signed zero: the old one-pass seed skipped these exact operations.
        if gain != 0. {
            seed.bias = finite(seed.bias - finite(gain * p.b1[j])?)?;
            for (i, r) in seed.residual.iter_mut().enumerate() {
                if !live(end) || !poll() {
                    return None;
                }
                *r = finite(*r - finite(gain * p.w_row(j)[i])?)?;
            }
        }
        seed.positive.push(j);
        seed.lambda.push(gain);
        Some(())
    }
}
/// The enclosing immutable Ticket prepays all cache, rejected pool entries,
/// ranking, seed fitting and payload. Cancellation is final at the caller.
pub(super) fn seed(
    p: &HullMarginProblem,
    point: &Point,
    head: usize,
    q: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Seed> {
    let d = p.input_dim;
    if head >= p.hidden_dim
        || q == 0
        || q > 32
        || d == 0
        || d > 128
        || p.w1.len() != p.hidden_dim.checked_mul(d)?
        || p.b1.len() != p.hidden_dim
        || p.lower.len() != d
        || p.upper.len() != d
        || point.values.len() != p.hidden_dim
    {
        return None;
    }
    let coordinates = Coordinates::new(p, head, end, poll)?;
    // Stable append-only records: the ranked16-byte pool moves only indices.
    // Comparator ties use ORIGINAL head indices, never feature allocation order.
    let mut features: Vec<Feature> = Vec::with_capacity(point.positive.len());
    let mut pool: Vec<(f64, usize)> = Vec::with_capacity(q);
    for &j in &point.positive {
        if !live(end) || !poll() {
            return None;
        }
        if *p.c.get(j)? <= 0. || pool.iter().any(|(_, k)| features[*k].head == j) {
            continue;
        }
        let (feature, score) = coordinates.feature(p, point, j, end, poll)?;
        if score <= 0. {
            continue;
        }
        let index = features.len();
        features.push(feature);
        let slot = pool
            .iter()
            .position(|&(old, k)| score > old || (score == old && j < features[k].head))
            .unwrap_or(pool.len());
        if slot < q {
            pool.insert(slot, (score, index));
            if pool.len() > q {
                pool.pop();
            }
        }
    }
    let mut seed = Seed {
        positive: Vec::with_capacity(q),
        lambda: Vec::with_capacity(q),
        residual: p.w_row(head).to_vec(),
        bias: p.b1[head],
    };
    for (_, index) in pool {
        if !live(end) || !poll() {
            return None;
        }
        coordinates.fit(p, &features[index], &mut seed, end, poll)?;
    }
    Some(seed)
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_dense_fused_tests.rs"]
mod tests;
