// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exclusive single-gradient OR independently priced four-branch tau advice.
use super::super::super::super::primal_advice::Point;
use super::*;
#[path = "single_hidden_hull_mip_signed_domination_branch_steering.rs"]
pub(super) mod steering;
pub(super) struct Planning {
    width: usize,
    optimizer: usize,
    checker: usize,
    old_sort: usize,
    extra_bytes: usize,
}
impl Planning {
    /// Scalar-only, hoisted once BEFORE the single bounded q quote loop.
    pub(super) fn new(p: &HullMarginProblem) -> Option<Self> {
        let d = p.input_dim;
        let width = d.checked_add(1)?;
        let levels = if d <= 1 {
            0
        } else {
            usize::BITS as usize - (d - 1).leading_zeros() as usize
        };
        let old_sort = d.checked_mul(levels)?.checked_mul(64)?;
        let optimizer = old_sort
            .checked_mul(4)?
            .checked_add(width.checked_mul(1344)?)?
            .checked_add(4096)?;
        let checker = width.checked_mul(1152)?.checked_add(128)?;
        let extra_bytes = width.checked_mul(192)?.checked_add(8192)?;
        Some(Self {
            width,
            optimizer,
            checker,
            old_sort,
            extra_bytes,
        })
    }
    pub(super) fn quote_base(
        &self,
        p: &HullMarginProblem,
        components: usize,
        capture: usize,
        base: dense::Quote,
    ) -> Option<Quote> {
        if base.negative_heads < 2
            || base
                .positive_limit
                .checked_add(p.input_dim)?
                .checked_add(2)?
                > 64
        {
            return None;
        }
        let nominal = base
            .positive_limit
            .checked_mul(self.width)?
            .checked_mul(64)?
            .checked_add(self.width.checked_mul(576)?)?
            .checked_add(base.positive_limit.checked_mul(64)?)?
            .checked_add(1024)?;
        let removed_old_tail = base
            .step_work
            .checked_add(self.old_sort)?
            .checked_add(base.alias_work)?;
        // Eligibility precedes BOTH routes. G already pays64q; the single
        // alternative must pay it too, independently of any alleged slack.
        let single = removed_old_tail.checked_add(base.positive_limit.checked_mul(64)?)?;
        let branch = nominal
            .checked_add(self.optimizer)?
            .checked_add(self.checker)?;
        let incremental = base
            .incremental_work
            .checked_sub(removed_old_tail)?
            .checked_add(single.max(branch))?;
        let total = capture.checked_add(incremental)?;
        let payload = base.bytes.checked_add(self.extra_bytes)?;
        (total <= resources::MAX_DOMINATION_WORK).then_some(Quote {
            source: std::ptr::from_ref(p) as usize,
            dimensions: (p.input_dim, p.hidden_dim),
            components,
            base,
            total,
            incremental,
            payload,
            removed_old_tail,
            single,
            branch,
            nominal,
            optimizer: self.optimizer,
            checker: self.checker,
        })
    }
}
pub(super) struct Quote {
    source: usize,
    dimensions: (usize, usize),
    components: usize,
    base: dense::Quote,
    total: usize,
    incremental: usize,
    payload: usize,
    pub removed_old_tail: usize,
    pub single: usize,
    pub branch: usize,
    pub nominal: usize,
    pub optimizer: usize,
    pub checker: usize,
}
impl Quote {
    pub(super) fn base(&self) -> &dense::Quote {
        &self.base
    }
    pub(super) fn total_work(&self) -> usize {
        self.total
    }
    /// Recheck both exclusive tails against the original paid base before debit.
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
        let expected = Planning::new(p)?.quote_base(p, components, capture, self.base.clone())?;
        (self.total_work() == expected.total_work()
            && self.incremental == expected.incremental
            && self.payload == expected.payload
            && self.removed_old_tail == expected.removed_old_tail
            && self.single == expected.single
            && self.branch == expected.branch
            && self.nominal == expected.nominal
            && self.optimizer == expected.optimizer
            && self.checker == expected.checker)
            .then_some(self.total_work())
    }
    pub(super) fn incremental_work(&self) -> usize {
        self.incremental
    }
    pub(super) fn bytes(&self) -> usize {
        self.payload
    }
    pub(super) fn matches(&self, p: &HullMarginProblem, c: usize) -> bool {
        self.source == std::ptr::from_ref(p) as usize
            && self.dimensions == (p.input_dim, p.hidden_dim)
            && self.components == c
    }
}
pub(super) struct Proposal {
    pub affine: group::kernel::Proposal,
    pub tau: [f64; 4],
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn clock(end: Instant, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    (live(end) && poll()).then_some(())
}
/// Serialize B ONCE from the exact final lambda. All float values are advice.
fn affines(
    p: &HullMarginProblem,
    a: &group::kernel::Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<([Vec<f64>; 2], [f64; 2], Vec<f64>, f64)> {
    let weights = [-*p.c.get(a.negative[0])?, -*p.c.get(a.negative[1])?];
    if weights.iter().any(|x| !x.is_finite() || *x <= 0.) {
        return None;
    }
    let bias = [
        finite(weights[0] * p.b1[a.negative[0]])?,
        finite(weights[1] * p.b1[a.negative[1]])?,
    ];
    let mut bb = 0.;
    for (&j, &l) in a.positive.iter().zip(&a.lambda) {
        clock(end, poll)?;
        bb = finite(bb + finite(l * p.b1[j])?)?;
    }
    let mut aa = [
        Vec::with_capacity(p.input_dim),
        Vec::with_capacity(p.input_dim),
    ];
    let mut b = Vec::with_capacity(p.input_dim);
    for i in 0..p.input_dim {
        clock(end, poll)?;
        for t in 0..2 {
            aa[t].push(finite(weights[t] * p.w_row(a.negative[t])[i])?);
        }
        let mut value = 0.;
        for (&j, &l) in a.positive.iter().zip(&a.lambda) {
            clock(end, poll)?;
            value = finite(value + finite(l * p.w_row(j)[i])?)?;
        }
        b.push(value);
    }
    Some((aa, bias, b, bb))
}
/// No second pool/fit/gradient. An interrupted or numerically failed optimizer
/// declines the pair; only a COMPLETED pass can retain its per-mask half bound.
pub(super) fn proposal(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    point: &Point,
    ticket: &resources::Ticket,
    advice: &dense::Advice,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Proposal> {
    let paid = ticket.branch_quote()?;
    let p = cuts.problem;
    let end = end
        .min(ticket.end)
        .min(row.authority())
        .min(row.budget.search_end)
        .min(cuts.deadline);
    if !paid.matches(p, row.plan.components.len())
        || advice.useful
        || advice.single.positive.len() > paid.base.positive_limit
        || !resources::begin_branch_tail(row, ticket)
    {
        return None;
    }
    clock(end, poll)?;
    let anchor = advice.single.head;
    let other = advice.partner?;
    if anchor == other {
        return None;
    }
    let scale = finite(-*p.c.get(anchor)?)?;
    if scale <= 0. {
        return None;
    }
    let lambda = advice
        .single
        .lambda
        .iter()
        .map(|a| finite(scale * a))
        .collect::<Option<Vec<_>>>()?;
    if !lambda.iter().any(|a| *a > 0.) {
        return None;
    }
    let mut affine = group::kernel::Proposal {
        negative: [anchor, other],
        positive: advice.single.positive.clone(),
        lambda,
        v: Vec::new(),
    };
    let (aa, bias, b, bb) = affines(p, &affine, end, poll)?;
    let mut half = [
        Vec::with_capacity(p.input_dim),
        Vec::with_capacity(p.input_dim),
    ];
    for i in 0..p.input_dim {
        clock(end, poll)?;
        for t in 0..2 {
            half[t].push(finite(aa[t][i] - finite(0.5 * b[i])?)?);
        }
    }
    let v0 = dense::direction(&half[0], finite(bias[0] - finite(0.5 * bb)?)?, p)?;
    let v1 = dense::direction(&half[1], finite(bias[1] - finite(0.5 * bb)?)?, p)?;
    affine.v = v0
        .iter()
        .zip(&v1)
        .map(|(a, b)| finite(a + b))
        .collect::<Option<Vec<_>>>()?;
    let problem = steering::Affines {
        a: [&aa[0], &aa[1]],
        bias,
        b: &b,
        b_bias: bb,
        v: &affine.v,
        lower: &p.lower,
        upper: &p.upper,
        point: &point.input,
    };
    let minimum = steering::minimize(&problem, end, poll)?;
    if !minimum.upper.is_finite() {
        return None;
    }
    let mut lhs = finite(
        finite(-p.c[anchor] * point.values[anchor])? + finite(-p.c[other] * point.values[other])?,
    )?;
    for (&j, &l) in affine.positive.iter().zip(&affine.lambda) {
        clock(end, poll)?;
        lhs = finite(lhs - finite(l * point.values[j])?)?;
    }
    clock(end, poll)?;
    {
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.domination;
        r.group_anchor = Some(anchor);
        r.group_partner = Some(other);
        r.group_nominal_masks = 4;
        r.group_nominal_lhs = Some(lhs);
        r.group_nominal_bound = Some(minimum.price);
        r.branch_tau_completed = 1;
        r.branch_tau_values = Some(minimum.tau);
        r.branch_tau_mask_prices = Some(minimum.mask_bounds);
        r.branch_tau_selected = minimum
            .tau
            .iter()
            .zip([0.0_f64, 0.5, 0.5, 1.])
            .any(|(a, b)| a.to_bits() != b.to_bits());
    }
    (lhs > minimum.price).then_some(Proposal {
        affine,
        tau: minimum.tau,
    })
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_branch_default_tests.rs"]
mod default_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_branch_test_fixture.rs"]
pub(super) mod test_fixture;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_branch_tests.rs"]
mod tests;
