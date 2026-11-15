// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One prepaid nominal coefficient step. Only the original support kernel proves rows.
use std::time::Instant;
const MAX_INPUTS: usize = 128;
const MAX_PARTNERS: usize = 32;
#[derive(Clone, Copy)]
pub(super) struct Head<'a> {
    pub weights: &'a [f64],
    pub bias: f64,
    pub value: f64,
}
pub(super) struct Candidate {
    pub lambda: Vec<f64>,
    pub slope: Vec<f64>,
    pub value: f64,
    pub attempted: bool,
    pub completed: bool,
    pub improved: bool,
}
struct Cache<'a> {
    lower: &'a [f64],
    upper: &'a [f64],
    point: &'a [f64],
    width: Vec<f64>,
    t: Vec<f64>,
    descending: Vec<usize>,
    ascending: Vec<usize>,
}
struct Chain {
    slope: Vec<f64>,
    moments: Vec<f64>,
    mass: f64,
    envelope: f64,
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn active(x: f64) -> f64 {
    if x > 0. {
        1.
    } else if x < 0. {
        0.
    } else {
        0.5
    }
}
fn live(end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    (Instant::now() < end && poll()).then_some(())
}
/// Separate from the old oriented order: each level reads/writes each index
/// once, <=D comparisons, no Entry/float arithmetic or input-sized reallocation.
fn merge_fixed_t(
    order: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
    t: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let n = order.len();
    let mut width = 1;
    while width < n {
        let mut start = 0;
        while start < n {
            let middle = (start + width).min(n);
            let end_at = (middle + width).min(n);
            let (mut left, mut right) = (start, middle);
            for slot in &mut scratch[start..end_at] {
                live(end, poll)?;
                let take_left =
                    right == end_at || (left != middle && t[order[left]] >= t[order[right]]);
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
            start = end_at;
        }
        std::mem::swap(order, scratch);
        width *= 2;
    }
    Some(())
}
impl<'a> Cache<'a> {
    /// Exactly ONE stable index merge sort of fixed original t. Zero residual
    /// coordinates stay in the cache: their means matter to the subgradient.
    fn new(
        lower: &'a [f64],
        upper: &'a [f64],
        point: &'a [f64],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        let n = lower.len();
        if n > MAX_INPUTS || upper.len() != n || point.len() != n {
            return None;
        }
        live(end, poll)?;
        let (mut width, mut t) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut descending: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            live(end, poll)?;
            if !lower[i].is_finite()
                || !upper[i].is_finite()
                || !point[i].is_finite()
                || lower[i] > upper[i]
                || point[i] < lower[i]
                || point[i] > upper[i]
            {
                return None;
            }
            let w = finite(upper[i] - lower[i])?;
            width.push(w);
            let normalized = if w == 0. {
                0.
            } else {
                finite(finite(point[i] - lower[i])? / w)?.clamp(0., 1.)
            };
            t.push(normalized);
            if w == 0. {
                continue;
            }
            descending.push(i);
        }
        // The same scratch allocation becomes the ascending stream afterward.
        let mut ascending = vec![0; descending.len()];
        merge_fixed_t(&mut descending, &mut ascending, &t, end, poll)?;
        ascending.clear();
        // Reverse equal-value GROUPS, preserving original-index order in ties.
        let mut end_at = descending.len();
        while end_at > 0 {
            live(end, poll)?;
            let mut start = end_at - 1;
            while start > 0 && t[descending[start - 1]] == t[descending[end_at - 1]] {
                start -= 1;
            }
            ascending.extend_from_slice(&descending[start..end_at]);
            end_at = start;
        }
        Some(Self {
            lower,
            upper,
            point,
            width,
            t,
            descending,
            ascending,
        })
    }
    fn theta(&self, c: &[f64], i: usize) -> f64 {
        if c[i] < 0. {
            1. - self.t[i]
        } else {
            self.t[i]
        }
    }
    /// Two filtered streams, one linear merge; no new residual-dependent sort.
    /// Distinct t may round to equal 1-t: those negative-stream ties inherit
    /// ascending t order. No parity with the old (u-x)/width order is claimed.
    fn order(&self, c: &[f64], end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<Vec<usize>> {
        let mut positive = self
            .descending
            .iter()
            .copied()
            .filter(|&i| c[i] >= 0.)
            .peekable();
        let mut negative = self
            .ascending
            .iter()
            .copied()
            .filter(|&i| c[i] < 0.)
            .peekable();
        let mut order = Vec::with_capacity(c.len());
        while positive.peek().is_some() || negative.peek().is_some() {
            live(end, poll)?;
            let take_positive = match (positive.peek(), negative.peek()) {
                (Some(&i), Some(&j)) => {
                    self.theta(c, i) > self.theta(c, j)
                        || (self.theta(c, i) == self.theta(c, j) && i < j)
                }
                (Some(_), None) => true,
                _ => false,
            };
            order.push(if take_positive {
                positive.next()?
            } else {
                negative.next()?
            });
        }
        Some(order)
    }
    fn chain(
        &self,
        c: &[f64],
        bias: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Chain> {
        if c.len() != self.width.len() || !bias.is_finite() {
            return None;
        }
        let mut value = bias;
        for (i, &coefficient) in c.iter().enumerate() {
            live(end, poll)?;
            let base = if coefficient < 0. {
                self.upper[i]
            } else {
                self.lower[i]
            };
            value = finite(value + finite(finite(coefficient)? * base)?)?;
        }
        let order = self.order(c, end, poll)?;
        let first = order.first().map_or(0., |&i| self.theta(c, i));
        let mut mass = finite((1. - first) * active(value))?;
        let mut envelope = finite((1. - first) * value.max(0.))?;
        let (mut slope, mut moments) = (vec![0.; c.len()], vec![0.; c.len()]);
        for (rank, &i) in order.iter().enumerate() {
            live(end, poll)?;
            moments[i] = mass; // Prefix mass BEFORE this coordinate flips.
            let increment = finite(c[i].abs() * self.width[i])?;
            if c[i] != 0. && increment <= 0. {
                return None;
            }
            let previous = value.max(0.);
            value = finite(value + increment)?;
            let delta = finite(value.max(0.) - previous)?;
            if delta < 0. {
                return None;
            }
            let sign = if c[i] < 0. { -1. } else { 1. };
            slope[i] = finite(sign * finite(delta / self.width[i])?)?;
            let next = order.get(rank + 1).map_or(0., |&j| self.theta(c, j));
            let weight = finite(self.theta(c, i) - next)?;
            if weight < 0. {
                return None;
            }
            mass = finite(mass + finite(weight * active(value))?)?;
            envelope = finite(envelope + finite(weight * value.max(0.))?)?;
        }
        self.finish_moments(c, mass, &mut moments, end, poll)?;
        Some(Chain {
            slope,
            moments,
            mass,
            envelope,
        })
    }
    fn finish_moments(
        &self,
        c: &[f64],
        mass: f64,
        moments: &mut [f64],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        for i in 0..c.len() {
            live(end, poll)?;
            let (base, sign) = if c[i] < 0. {
                (self.upper[i], -1.)
            } else {
                (self.lower[i], 1.)
            };
            let suffix = if self.width[i] == 0. {
                0.
            } else {
                finite(mass - moments[i])?
            };
            moments[i] = finite(finite(base * mass)? + finite(sign * self.width[i] * suffix)?)?;
        }
        Some(())
    }
    fn plane_value(
        &self,
        c: &[f64],
        bias: f64,
        v: &[f64],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        let (mut zero, mut positive, mut at) = (0., bias, 0.);
        for i in 0..c.len() {
            live(end, poll)?;
            let z = finite(-v[i])?;
            let r = finite(c[i] - v[i])?;
            zero = finite(zero + finite(z * self.lower[i])?.max(finite(z * self.upper[i])?))?;
            positive =
                finite(positive + finite(r * self.lower[i])?.max(finite(r * self.upper[i])?))?;
            at = finite(at + finite(v[i] * self.point[i])?)?;
        }
        finite(zero.max(positive) + at)
    }
}
fn residual(
    target: Head<'_>,
    partners: &[Head<'_>],
    lambda: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(Vec<f64>, f64)> {
    let mut c = target.weights.to_vec();
    let mut bias = finite(target.bias)?;
    for (&weight, partner) in lambda.iter().zip(partners) {
        live(end, poll)?;
        if !weight.is_finite()
            || !(0. ..=256.).contains(&weight)
            || partner.weights.len() != c.len()
            || !partner.value.is_finite()
        {
            return None;
        }
        bias = finite(bias - finite(weight * finite(partner.bias)?)?)?;
        for (value, &coefficient) in c.iter_mut().zip(partner.weights) {
            live(end, poll)?;
            *value = finite(finite(*value)? - finite(weight * finite(coefficient)?)?)?;
        }
    }
    Some((c, bias))
}
fn head_price(partners: &[Head<'_>], lambda: &[f64]) -> Option<f64> {
    lambda
        .iter()
        .zip(partners)
        .try_fold(0., |sum, (&a, head)| finite(sum + finite(a * head.value)?))
}
fn gradient(
    chain: &Chain,
    partners: &[Head<'_>],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<f64>> {
    partners
        .iter()
        .map(|head| {
            live(end, poll)?;
            let mut expected = finite(head.bias * chain.mass)?;
            for (&coefficient, &moment) in head.weights.iter().zip(&chain.moments) {
                live(end, poll)?;
                expected = finite(expected + finite(coefficient * moment)?)?;
            }
            finite(head.value - expected)
        })
        .collect()
}
fn trial(seed: &[f64], gradient: &[f64], envelope: f64, target: f64) -> Option<Vec<f64>> {
    let norm = gradient
        .iter()
        .try_fold(0., |sum, &g| finite(sum + finite(g * g)?))?;
    if envelope <= target || norm <= 0. {
        return None;
    }
    let eta = finite(finite(1.5 * finite(envelope - target)?)? / norm)?.min(4.);
    seed.iter()
        .zip(gradient)
        .map(|(&a, &g)| finite(a - finite(eta * g)?).map(|v| v.clamp(0., 256.)))
        .collect()
}
fn retain(result: Candidate, end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<Candidate> {
    live(end, poll)?;
    Some(result)
}
fn improve_once(
    cache: &Cache<'_>,
    initial: &Chain,
    target: Head<'_>,
    partners: &[Head<'_>],
    seed: &[f64],
    envelope: f64,
    result: &mut Candidate,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let g = gradient(initial, partners, end, poll)?;
    let lambda = trial(seed, &g, envelope, target.value)?;
    let (c, bias) = residual(target, partners, &lambda, end, poll)?;
    let updated = cache.chain(&c, bias, end, poll)?;
    let value = finite(
        head_price(partners, &lambda)? + cache.plane_value(&c, bias, &updated.slope, end, poll)?,
    )?;
    result.completed = true;
    if value < result.value {
        result.lambda = lambda;
        result.slope = updated.slope;
        result.value = value;
        result.improved = true;
    }
    Some(())
}
/// All work/payload, including refusal, must be prepaid by the dense Ticket.
/// One seed chain and at most ONE projected trial use the same original end.
/// Outside-box/nonfinite hints decline; no point repair or witness is produced.
pub(super) fn improve(
    target: Head<'_>,
    partners: &[Head<'_>],
    seed: &[f64],
    lower: &[f64],
    upper: &[f64],
    point: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Candidate> {
    if partners.is_empty()
        || partners.len() > MAX_PARTNERS
        || seed.len() != partners.len()
        || target.weights.len() != lower.len()
        || !target.value.is_finite()
    {
        return None;
    }
    // A cancellation is final even if a caller callback later returns true.
    let mut cancelled = false;
    let mut observed_poll = || {
        if cancelled {
            return false;
        }
        let value = poll();
        cancelled = !value;
        value
    };
    let poll: &mut dyn FnMut() -> bool = &mut observed_poll;
    let cache = Cache::new(lower, upper, point, end, poll)?;
    let (c, bias) = residual(target, partners, seed, end, poll)?;
    let initial = cache.chain(&c, bias, end, poll)?;
    let positive = head_price(partners, seed)?;
    let value = finite(positive + cache.plane_value(&c, bias, &initial.slope, end, poll)?)?;
    let mut result = Candidate {
        lambda: seed.to_vec(),
        slope: initial.slope.clone(),
        value,
        attempted: false,
        completed: false,
        improved: false,
    };
    let envelope = finite(positive + initial.envelope)?;
    if envelope <= target.value {
        return retain(result, end, poll);
    }
    result.attempted = true;
    let _ = improve_once(
        &cache,
        &initial,
        target,
        partners,
        seed,
        envelope,
        &mut result,
        end,
        poll,
    );
    retain(result, end, poll)
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_priced_tests.rs"]
mod tests;
