// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent common-slope signed support, never an output-margin certificate.
use super::*;

#[derive(Clone)]
pub(super) struct PositiveTerms(pub(super) Vec<(usize, f64)>);

pub(super) struct PositiveProof {
    terms: PositiveTerms,
    bits: Vec<u64>,
}

#[derive(Default)]
pub(super) struct AttemptReport {
    pub(super) attempted: bool,
    pub(super) positive_heads: usize,
    pub(super) negative_heads: usize,
    pub(super) masks: usize,
    pub(super) cells: usize,
    pub(super) lower: Option<f64>,
    pub(super) admitted: bool,
}

impl AttemptReport {
    pub(super) fn emit(&self) {
        let bits = self
            .lower
            .map_or_else(|| "none".to_owned(), |v| format!("{:016x}", v.to_bits()));
        eprintln!("[input-head-directional-signed] attempted={} positive_heads={} negative_heads={} completed_masks={} certificate_work_cells={} certified={} lower_bits={} admitted={}",
            usize::from(self.attempted), self.positive_heads, self.negative_heads, self.masks, self.cells,
            usize::from(self.lower.is_some()), bits, usize::from(self.admitted));
    }
}

pub(super) fn point(value: I) -> Option<f64> {
    let point = 0.5 * value.lo + 0.5 * value.hi;
    (value.valid() && good(point)).then_some(point)
}

fn valid_terms(source: &Source, terms: &[(usize, f64)], maximum: usize) -> bool {
    !terms.is_empty()
        && terms.len() <= maximum
        && terms.iter().enumerate().all(|(at, &(j, a))| {
            j < source.spec.heads && good(a) && a > 0.0 && (at == 0 || terms[at - 1].0 < j)
        })
}

fn valid_geometry(
    source: &Source,
    negative: &Direction,
    positive: &PositiveTerms,
    eta: &[(usize, f64)],
) -> bool {
    valid_terms(source, &negative.heads, MAX_SUPPORT)
        && valid_terms(source, &positive.0, source.spec.heads)
        && valid_eta(source, eta)
        && positive.0.iter().all(|&(j, _)| {
            negative
                .heads
                .binary_search_by_key(&j, |&(k, _)| k)
                .is_err()
        })
}

impl PositiveProof {
    pub(super) fn terms(&self) -> &[(usize, f64)] {
        &self.terms.0
    }

    pub(super) fn valid(
        &self,
        source: &Source,
        negative: &Direction,
        eta: &[(usize, f64)],
    ) -> bool {
        if !valid_geometry(source, negative, &self.terms, eta) {
            return false;
        }
        let lengths = [
            negative.heads.len() as u64,
            self.terms.0.len() as u64,
            eta.len() as u64,
        ];
        let expected = lengths.into_iter().chain(
            negative
                .heads
                .iter()
                .chain(&self.terms.0)
                .chain(eta)
                .flat_map(|&(j, a)| [j as u64, a.to_bits()]),
        );
        self.bits.iter().copied().eq(expected)
    }
}

fn aggregate(source: &Source, positive: &PositiveTerms, budget: &mut Budget) -> Option<Affine> {
    let d = source.spec.inputs;
    budget.charge(d)?;
    let mut result = Affine::zero(d);
    for &(head, alpha) in &positive.0 {
        let error = source.first_error[head];
        if !good(error) || error < 0.0 {
            return None;
        }
        budget.charge(d.checked_mul(2)?.checked_add(4)?)?;
        let a = I::p(alpha);
        result.b = result
            .b
            .add(a.mul(I::p(f64::from(source.spec.first_bias[head])).sub(I::p(error))?)?)?;
        for j in 0..d {
            if j & 31 == 0 {
                budget.charge(0)?;
            }
            result.a[j] = result.a[j].add(a.mul(I::p(f64::from(
                source.spec.first[j * source.spec.heads + head],
            )))?)?;
        }
    }
    Some(result)
}

fn negative_form(
    source: &Source,
    negative: &Direction,
    eta: &[(usize, f64)],
    mask: usize,
    budget: &mut Budget,
) -> Option<(Affine, f64)> {
    let d = source.spec.inputs;
    budget.charge(d.checked_add(eta.len())?)?;
    let mut form = Affine::zero(d);
    for &(j, a) in eta {
        form.a[j] = I::p(a);
    }
    let mut mass = 0.0;
    for (bit, &(head, tau)) in negative.heads.iter().enumerate() {
        if mask & (1 << bit) == 0 {
            continue;
        }
        let error = source.first_error[head];
        if !good(error) || error < 0.0 {
            return None;
        }
        budget.charge(d.checked_mul(2)?.checked_add(4)?)?;
        mass += tau;
        let a = I::p(-tau);
        form.b = form
            .b
            .add(a.mul(I::p(f64::from(source.spec.first_bias[head])).add(I::p(error))?)?)?;
        for j in 0..d {
            if j & 31 == 0 {
                budget.charge(0)?;
            }
            form.a[j] = form.a[j].add(a.mul(I::p(f64::from(
                source.spec.first[j * source.spec.heads + head],
            )))?)?;
        }
    }
    good(mass).then_some((form, mass))
}

/// Midpoint one-dimensional concave box objective: search only. A finite
/// chosen theta is replayed independently below; no search score is a bound.
fn breakpoint(
    form: &Affine,
    positive: &Affine,
    source: &Source,
    budget: &mut Budget,
) -> Option<f64> {
    let d = source.spec.inputs;
    let log = usize::BITS as usize - d.max(1).leading_zeros() as usize;
    budget.charge(d.checked_mul(log.checked_add(6)?)?.checked_mul(2)?)?;
    let mut points = Vec::with_capacity(d);
    let mut derivative = point(positive.b)?;
    for j in 0..d {
        if j & 31 == 0 {
            budget.charge(0)?;
        }
        let a = point(form.a[j])?;
        let p = point(positive.a[j])?;
        derivative += p * if a < 0.0 || (a == 0.0 && p < 0.0) {
            source.hi[j]
        } else {
            source.lo[j]
        };
        let theta = -a / p;
        let drop = p.abs() * (source.hi[j] - source.lo[j]);
        if good(theta) && theta > 0.0 && theta < 1.0 && good(drop) {
            points.push((theta, drop));
        }
    }
    if !good(derivative) {
        return None;
    }
    if derivative <= 0.0 {
        return Some(0.0);
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (theta, drop) in points {
        derivative -= drop;
        if derivative <= 0.0 {
            return Some(theta);
        }
    }
    Some(1.0)
}

fn replay_theta(
    source: &Source,
    form: &Affine,
    positive: &Affine,
    theta: f64,
    budget: &mut Budget,
) -> Option<f64> {
    if !good(theta) || !(0.0..=1.0).contains(&theta) {
        return None;
    }
    budget.charge(source.spec.inputs.checked_mul(4)?.checked_add(3)?)?;
    let t = I::p(theta);
    let mut value = form.b.add(t.mul(positive.b)?)?;
    for j in 0..source.spec.inputs {
        if j & 31 == 0 {
            budget.charge(0)?;
        }
        let coefficient = form.a[j].add(t.mul(positive.a[j])?)?;
        value = value.add(coefficient.mul(I {
            lo: source.lo[j],
            hi: source.hi[j],
        })?)?;
    }
    (value.valid() && live(budget.deadline)).then_some(value.lo)
}

fn independent_box(
    source: &Source,
    negative: &Direction,
    positive: &PositiveTerms,
    eta: &[(usize, f64)],
    budget: &mut Budget,
) -> Option<f64> {
    budget.charge(
        eta.len()
            .checked_add(negative.heads.len())?
            .checked_add(positive.0.len())?,
    )?;
    let mut floor = I::p(0.0);
    for &(j, a) in eta {
        floor = floor.add(I::p(a).mul(I {
            lo: source.lo[j],
            hi: source.hi[j],
        })?)?;
    }
    for &(j, a) in &positive.0 {
        floor = floor.add(I::p(a).mul(I::p(source.y[j].lo.max(0.0)))?)?;
    }
    for &(j, a) in &negative.heads {
        floor = floor.add(I::p(-a).mul(I::p(source.y[j].hi.max(0.0)))?)?;
    }
    Some(floor.lo)
}

/// For every negative algebraic mask S, alpha*r >= theta_S*alpha*(t-E1).
/// Every mask, including empty, must complete. All original coordinates are
/// contracted; output bias/Eout/q and other portfolio rows are not premises.
pub(super) fn certify<'a>(
    source: &'a Source,
    negative: Direction,
    positive: PositiveTerms,
    eta: Vec<(usize, f64)>,
    budget: &mut Budget,
) -> Option<CertifiedDirection<'a>> {
    let d = source.spec.inputs;
    let h = source.spec.heads;
    budget.charge(h.checked_add(d)?)?;
    if !valid_geometry(source, &negative, &positive, &eta)
        || source.lo.len() != d
        || source.hi.len() != d
        || source.spec.first.len() != d.checked_mul(h)?
        || source.spec.first_bias.len() != h
        || source.first_error.len() != h
        || source.y.len() != h
    {
        return None;
    }
    let count = 1usize.checked_shl(negative.heads.len() as u32)?;
    if budget.masks.checked_add(count)? > MAX_MASKS {
        return None;
    }
    let p = aggregate(source, &positive, budget)?;
    let total: f64 = negative.heads.iter().map(|&(_, a)| a).sum();
    if !good(total) || total <= 0.0 {
        return None;
    }
    let mut lower = f64::INFINITY;
    for mask in 0..count {
        let (form, selected) = negative_form(source, &negative, &eta, mask, budget)?;
        let ratio = (selected / total).clamp(0.0, 1.0);
        let chosen = breakpoint(&form, &p, source, budget)?;
        let mut best = f64::NEG_INFINITY;
        for theta in [0.0, 1.0, ratio, chosen] {
            best = best.max(replay_theta(source, &form, &p, theta, budget)?);
        }
        lower = lower.min(best);
        budget.complete_mask()?;
    }
    finish(source, negative, positive, eta, lower, budget)
}

fn finish<'a>(
    source: &'a Source,
    negative: Direction,
    positive: PositiveTerms,
    eta: Vec<(usize, f64)>,
    lower: f64,
    budget: &mut Budget,
) -> Option<CertifiedDirection<'a>> {
    let deadline = budget.deadline;
    finish_at(source, negative, positive, eta, lower, budget, deadline)
}

fn finish_at<'a>(
    source: &'a Source,
    negative: Direction,
    positive: PositiveTerms,
    eta: Vec<(usize, f64)>,
    lower: f64,
    budget: &mut Budget,
    deadline: Instant,
) -> Option<CertifiedDirection<'a>> {
    budget.charge(0)?;
    if !live(deadline) {
        return None;
    }
    if !good(lower) || lower <= independent_box(source, &negative, &positive, &eta, budget)? {
        return None;
    }
    let size = negative
        .heads
        .len()
        .checked_add(positive.0.len())?
        .checked_add(eta.len())?
        .checked_mul(2)?
        .checked_add(3)?;
    budget.charge(size)?;
    let mut bits = Vec::with_capacity(size);
    bits.extend([
        negative.heads.len() as u64,
        positive.0.len() as u64,
        eta.len() as u64,
    ]);
    for &(j, a) in negative.heads.iter().chain(&positive.0).chain(&eta) {
        bits.extend([j as u64, a.to_bits()]);
    }
    Some(CertifiedDirection {
        source,
        direction: negative,
        eta,
        lower,
        deadline,
        geometry: GeometryKind::Signed(PositiveProof {
            terms: positive,
            bits,
        }),
    })
}

#[path = "input_head_directional_signed_gray.rs"]
mod gray;
#[cfg(test)]
pub(super) use gray::certify as certify_gray;
pub(super) use gray::work_bound as gray_work_bound;
pub(super) use gray::{certify_protected as certify_gray_protected, Attempt as GrayAttempt};
#[cfg(test)]
pub(super) use gray::{test_form as gray_form, test_theta as gray_theta};

#[cfg(test)]
#[path = "input_head_directional_signed_tests.rs"]
mod tests;
