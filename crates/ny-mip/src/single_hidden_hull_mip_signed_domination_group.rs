// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One paid K2 nominal fallback. All final constants come from fresh Source.
use super::super::super::super::primal_advice::Point;
use super::*;
#[path = "single_hidden_hull_mip_signed_domination_group_kernel.rs"]
pub(super) mod kernel;
const K: usize = 2;
const MAX_TERMS: usize = 64;
pub(super) struct Quote {
    source: usize,
    dimensions: (usize, usize),
    components: usize,
    base: dense::Quote,
    total: usize,
    incremental: usize,
    payload: usize,
    nominal_work: usize,
    checker_tail: usize,
}
impl Quote {
    pub(super) fn base(&self) -> &dense::Quote {
        &self.base
    }
    pub(super) fn positive_limit(&self) -> usize {
        self.base.positive_limit
    }
    pub(super) fn total_work(&self) -> usize {
        self.total
    }
    /// Recheck the scalar producer formula before its selected debit.
    pub(super) fn checked_total(
        &self,
        p: &HullMarginProblem,
        components: usize,
        capture: usize,
    ) -> Option<usize> {
        if !self.matches(p, components) {
            return None;
        }
        self.base.checked_total(capture)?;
        let expected = quote(p, components, capture, self.base.clone())?;
        (self.total_work() == expected.total_work()
            && self.incremental == expected.incremental
            && self.payload == expected.payload
            && self.nominal_work == expected.nominal_work
            && self.checker_tail == expected.checker_tail)
            .then_some(self.total_work())
    }
    pub(super) fn incremental_work(&self) -> usize {
        self.incremental
    }
    pub(super) fn bytes(&self) -> usize {
        self.payload
    }
    pub(super) fn nominal_work(&self) -> usize {
        self.nominal_work
    }
    pub(super) fn checker_tail(&self) -> usize {
        self.checker_tail
    }
    fn matches(&self, p: &HullMarginProblem, components: usize) -> bool {
        self.source == std::ptr::from_ref(p) as usize
            && self.dimensions == (p.input_dim, p.hidden_dim)
            && self.components == components
    }
}
pub(super) fn quote(
    p: &HullMarginProblem,
    components: usize,
    capture: usize,
    base: dense::Quote,
) -> Option<Quote> {
    if base.negative_heads < 2
        || base
            .positive_limit
            .checked_add(p.input_dim)?
            .checked_add(K)?
            > MAX_TERMS
    {
        return None;
    }
    let w = p.input_dim.checked_add(1)?;
    let nominal_work = base
        .positive_limit
        .checked_mul(w)?
        .checked_mul(64)?
        .checked_add(w.checked_mul(576)?)?
        .checked_add(base.positive_limit.checked_mul(64)?)?
        .checked_add(1024)?;
    // The common single B/support/export fee remains in base. Reserve either
    // its complete alias tail OR K2's extra residual/four-mask/export work.
    let checker_tail = base.alias_work.max(w.checked_mul(1152)?.checked_add(128)?);
    let incremental = base
        .incremental_work
        .checked_sub(base.alias_work)?
        .checked_add(checker_tail)?
        .checked_add(nominal_work)?;
    let total = capture.checked_add(incremental)?;
    let payload = base
        .bytes
        .checked_add(w.checked_mul(128)?)?
        .checked_add(4096)?;
    (total <= resources::MAX_DOMINATION_WORK).then_some(Quote {
        source: std::ptr::from_ref(p) as usize,
        dimensions: (p.input_dim, p.hidden_dim),
        components,
        base,
        total,
        incremental,
        payload,
        nominal_work,
        checker_tail,
    })
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn weight(p: &HullMarginProblem, j: usize) -> Option<f64> {
    let w = -*p.c.get(j)?;
    (w.is_finite() && w > 0.).then_some(w)
}
fn check_clock(end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    (live(end) && poll()).then_some(())
}
/// Build B from EXACTLY the finite serialized coefficients, not subtraction
/// of cached aggregate/residual values. This float construction owns no H.
fn residuals(
    p: &HullMarginProblem,
    a: &kernel::Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<([Vec<f64>; K], [f64; K])> {
    let weights = [weight(p, a.negative[0])?, weight(p, a.negative[1])?];
    let mut b = 0.;
    for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
        check_clock(end, poll)?;
        b = finite(b + finite(lambda * p.b1[j])?)?;
    }
    let biases = [
        finite(finite(weights[0] * p.b1[a.negative[0]])? - finite(0.5 * b)?)?,
        finite(finite(weights[1] * p.b1[a.negative[1]])? - finite(0.5 * b)?)?,
    ];
    let mut dirs = [
        Vec::with_capacity(p.input_dim),
        Vec::with_capacity(p.input_dim),
    ];
    for i in 0..p.input_dim {
        check_clock(end, poll)?;
        let mut b = 0.;
        for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
            check_clock(end, poll)?;
            b = finite(b + finite(lambda * p.w_row(j)[i])?)?;
        }
        let half = finite(0.5 * b)?;
        for t in 0..K {
            dirs[t].push(finite(
                finite(weights[t] * p.w_row(a.negative[t])[i])? - half,
            )?);
        }
    }
    Some((dirs, biases))
}
/// Joint four-mask support AT THE COMMON direction. Summing two separately
/// optimized/scalar prices can be strictly weaker and incorrectly filter a
/// useful joint candidate. This remains nominal; the kernel redoes all masks.
fn price(
    p: &HullMarginProblem,
    dirs: &[Vec<f64>; K],
    bias: &[f64; K],
    v: &[f64],
    x: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let mut h = f64::NEG_INFINITY;
    for mask in 0..(1usize << K) {
        check_clock(end, poll)?;
        let mut value = 0.;
        for t in 0..K {
            if mask & (1 << t) != 0 {
                value = finite(value + bias[t])?;
            }
        }
        for i in 0..p.input_dim {
            check_clock(end, poll)?;
            let mut c = -v[i];
            for t in 0..K {
                if mask & (1 << t) != 0 {
                    c = finite(c + dirs[t][i])?;
                }
            }
            value = finite(value + finite(c * p.lower[i])?.max(finite(c * p.upper[i])?))?;
        }
        h = h.max(value);
    }
    let mut at = 0.;
    for (&a, &b) in v.iter().zip(x) {
        check_clock(end, poll)?;
        at = finite(at + finite(a * b)?)?;
    }
    finite(h + at)
}
/// Called only after the canonical winner was found nonviolating under ONE
/// paid advice capability. A useful single winner never enters this fallback. No second pool scan, fit,
/// priced step, sort, optimizer mask, LP or fresh deadline occurs here.
pub(super) fn proposal(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    point: &Point,
    paid: &Quote,
    advice: &dense::Advice,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<kernel::Proposal> {
    let p = cuts.problem;
    let end = end
        .min(row.authority())
        .min(row.budget.search_end)
        .min(cuts.deadline);
    if !paid.matches(p, row.plan.components.len())
        || advice.useful
        || advice.single.positive.len() > paid.positive_limit()
    {
        return None;
    }
    {
        let mut c = row.budget.state.lock().ok()?;
        c.component_rows.domination.group_nominal_attempts = 1;
    }
    check_clock(end, poll)?;
    let other = advice.partner?;
    let anchor = advice.single.head;
    if anchor == other {
        return None;
    }
    {
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.domination;
        r.group_anchor = Some(anchor);
        r.group_partner = Some(other);
    }
    let scale = weight(p, anchor)?;
    let lambda = advice
        .single
        .lambda
        .iter()
        .map(|a| finite(scale * a))
        .collect::<Option<Vec<_>>>()?;
    if !lambda.iter().any(|a| *a > 0.) {
        return None;
    }
    let mut candidate = kernel::Proposal {
        negative: [anchor, other],
        positive: advice.single.positive.clone(),
        lambda,
        v: Vec::new(),
    };
    let (dirs, biases) = residuals(p, &candidate, end, poll)?;
    let v0 = dense::direction(&dirs[0], biases[0], p)?;
    let v1 = dense::direction(&dirs[1], biases[1], p)?;
    candidate.v = v0
        .iter()
        .zip(&v1)
        .map(|(a, b)| finite(a + b))
        .collect::<Option<Vec<_>>>()?;
    let bound = price(p, &dirs, &biases, &candidate.v, &point.input, end, poll)?;
    let mut lhs = finite(
        finite(weight(p, anchor)? * point.values[anchor])?
            + finite(weight(p, other)? * point.values[other])?,
    )?;
    for (&j, &a) in candidate.positive.iter().zip(&candidate.lambda) {
        check_clock(end, poll)?;
        lhs = finite(lhs - finite(a * point.values[j])?)?;
    }
    check_clock(end, poll)?;
    {
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.domination;
        r.group_nominal_masks = 4;
        r.group_nominal_lhs = Some(lhs);
        r.group_nominal_bound = Some(bound);
    }
    (lhs > bound).then_some(candidate)
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_group_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_group_default_tests.rs"]
mod default_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_group_test_fixture.rs"]
mod test_fixture;
