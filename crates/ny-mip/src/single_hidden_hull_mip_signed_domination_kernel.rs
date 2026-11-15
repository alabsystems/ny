// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fresh original-affine support; nominal residual coefficients are never Source.
use super::super::certificate::I;
use super::*;
#[derive(Debug, Clone)]
pub(super) struct Proposal {
    pub head: usize,
    pub positive: Vec<usize>,
    pub lambda: Vec<f64>,
    pub v: Vec<f64>,
}
#[derive(Debug)]
pub(super) struct Checked {
    pub proposal: Proposal,
    pub upper: f64,
    pub aliases: Vec<usize>,
    pub alias_limit: usize,
    pub source: usize,
    pub ledger: Arc<std::sync::Mutex<Counts>>,
    pub witness: Arc<RetainedWords>,
    pub signature: Arc<RetainedWords>,
    pub bits: Vec<u64>,
    pub end: Instant,
    accepted_end: Instant,
}
fn bits(p: &Proposal, upper: f64, aliases: &[usize], limit: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(6 + 2 * p.positive.len() + p.v.len() + aliases.len());
    out.extend([p.head as u64, p.positive.len() as u64, p.v.len() as u64]);
    out.extend(p.positive.iter().map(|j| *j as u64));
    out.extend(p.lambda.iter().chain(&p.v).map(|x| x.to_bits()));
    out.extend([upper.to_bits(), limit as u64, aliases.len() as u64]);
    out.extend(aliases.iter().map(|j| *j as u64));
    out
}
fn shape(p: &HullMarginProblem, a: &Proposal) -> Option<()> {
    if a.head >= p.hidden_dim
        || !(1..=32).contains(&a.positive.len())
        || a.positive.len() != a.lambda.len()
        || a.v.len() != p.input_dim
        || p.w1.len() != p.hidden_dim.checked_mul(p.input_dim)?
        || p.b1.len() != p.hidden_dim
        || p.lower.len() != p.input_dim
        || p.upper.len() != p.input_dim
        || a.v.iter().any(|v| !v.is_finite())
        || a.lambda.iter().any(|v| !v.is_finite() || *v < 0.)
        || !a.lambda.iter().any(|v| *v > 0.)
    {
        return None;
    }
    for (i, &j) in a.positive.iter().enumerate() {
        if j >= p.hidden_dim || j == a.head || a.positive[..i].contains(&j) {
            return None;
        }
    }
    Some(())
}
fn maximum(a: I, l: f64, u: f64) -> Option<f64> {
    if !l.is_finite() || !u.is_finite() || l > u {
        return None;
    }
    if a.lo == 0. && a.hi == 0. {
        return Some(0.);
    }
    finite(
        mul_up(a.lo, l)
            .max(mul_up(a.lo, u))
            .max(mul_up(a.hi, l))
            .max(mul_up(a.hi, u)),
    )
}
fn sum_upper(a: f64, b: f64) -> Option<f64> {
    if b == 0. {
        finite(a)
    } else if a == 0. {
        finite(b)
    } else {
        finite(add_up(a, b))
    }
}
/// H=max(sup -v.x, sup (a-sum lambda*b-v.x)). Both branches use the
/// ORIGINAL signed box, including the complete bias and every input coordinate.
pub(super) fn upper(
    p: &HullMarginProblem,
    a: &Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    shape(p, a)?;
    require_hull_environment().ok()?;
    if !live(end) || !poll() {
        return None;
    }
    let mut bias = I::point(p.b1[a.head])?;
    for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
        if !live(end) || !poll() {
            return None;
        }
        bias = bias.sub(I::point(p.b1[j])?.scale(lambda)?)?;
    }
    let (mut zero, mut active) = (0., bias.hi);
    for i in 0..p.input_dim {
        if !live(end) || !poll() {
            return None;
        }
        let mut coefficient = I::point(p.w_row(a.head)[i])?;
        for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
            coefficient = coefficient.sub(I::point(p.w_row(j)[i])?.scale(lambda)?)?;
        }
        coefficient = coefficient.sub(I::point(a.v[i])?)?;
        active = sum_upper(active, maximum(coefficient, p.lower[i], p.upper[i])?)?;
        zero = sum_upper(zero, maximum(I::point(-a.v[i])?, p.lower[i], p.upper[i])?)?;
    }
    require_hull_environment().ok()?;
    (live(end) && poll())
        .then_some(zero.max(active))
        .and_then(finite)
}
pub(super) fn check(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    ticket: &resources::Ticket,
    proposal: Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    check_inner(row, cuts, ticket, proposal, None, end, poll)
}
/// Exact-affine reuse is admitted only with the immutable captured-negative
/// quota. It shares the canonical original support attempt and checker slot.
pub(super) fn check_with_aliases(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    ticket: &resources::Ticket,
    proposal: Proposal,
    negative: &[usize],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    check_inner(row, cuts, ticket, proposal, Some(negative), end, poll)
}
fn check_inner(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    ticket: &resources::Ticket,
    proposal: Proposal,
    negative: Option<&[usize]>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    let end = resources::support_attempt(row, ticket, end)?;
    let p = cuts.problem;
    if proposal.positive.len() > ticket.support_limit() {
        return None;
    }
    if row.source != std::ptr::from_ref(p) as usize
        || row.budget.source != row.source
        || p.c.get(proposal.head)?.partial_cmp(&0.) != Some(std::cmp::Ordering::Less)
        || proposal
            .positive
            .iter()
            .any(|&j| p.c.get(j).is_none_or(|v| *v <= 0.))
    {
        return None;
    }
    let upper = upper(p, &proposal, end, poll)?;
    let alias_limit = if negative.is_some() {
        ticket.alias_limit()
    } else {
        0
    };
    let aliases = if let Some(negative) = negative {
        alias::collect(p, proposal.head, negative, alias_limit, end, poll)?
    } else {
        vec![proposal.head]
    };
    let bits = bits(&proposal, upper, &aliases, alias_limit);
    let mut c = row.budget.state.lock().ok()?;
    c.component_rows.domination.completed_supports += 1;
    if alias_limit > 0 {
        c.component_rows.domination.checked_alias_rows += aliases.len();
    }
    Some(Checked {
        proposal,
        upper,
        aliases,
        alias_limit,
        source: row.source,
        ledger: Arc::clone(&row.budget.state),
        witness: Arc::clone(&row.witness),
        signature: Arc::clone(&row.signature),
        bits,
        end,
        accepted_end: end,
    })
}
impl Checked {
    pub(super) fn owned(&self, row: &SignedMajorant) -> bool {
        self.source == row.source
            && self.end == self.accepted_end
            && self.end <= row.authority()
            && self.end <= row.budget.search_end
            && self.upper.is_finite()
            && self.alias_limit <= row.budget.dimensions.1
            && !self.aliases.is_empty()
            && self.aliases.len() <= row.budget.dimensions.1
            && self.aliases.len() <= self.alias_limit.max(1)
            && self.aliases.windows(2).all(|pair| pair[0] < pair[1])
            && self.aliases.contains(&self.proposal.head)
            && (self.alias_limit > 0 || self.aliases == [self.proposal.head])
            && Arc::ptr_eq(&self.ledger, &row.budget.state)
            && Arc::ptr_eq(&self.witness, &row.witness)
            && Arc::ptr_eq(&self.signature, &row.signature)
            && self.bits == bits(&self.proposal, self.upper, &self.aliases, self.alias_limit)
    }
}
fn canonical_row(
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    token: &Checked,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(RowSpec, usize)> {
    shape(cuts.problem, &token.proposal)?;
    let a = &token.proposal;
    let mut coeffs = Vec::with_capacity(a.positive.len() + a.v.len() + 1);
    coeffs.push((binding.head_column(a.head)?.0, 1.));
    let mut row_heads = 1usize;
    for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
        if !live(end) || !poll() {
            return None;
        }
        if lambda != 0. {
            coeffs.push((binding.head_column(j)?.0, -lambda));
            row_heads += 1;
        }
    }
    for (i, &v) in a.v.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        if v != 0. {
            coeffs.push((parts.input_vars.get(i)?.0, -v));
        }
    }
    let canonical = RowSpec {
        lb: f64::NEG_INFINITY,
        ub: token.upper,
        coeffs,
    };
    Some((canonical, row_heads))
}
pub(super) fn export(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    checked: &[Checked],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    require_hull_environment().ok()?;
    if !binding.matches(cuts, parts) || row.source != std::ptr::from_ref(cuts.problem) as usize {
        return None;
    }
    {
        let mut c = row.budget.state.lock().ok()?;
        if c.replay_attempts != 1 || c.source_replays != 1 {
            return None;
        }
        let r = &mut c.component_rows.domination;
        if !r.admitted || r.export_attempts != 0 || checked.len() > r.completed_supports {
            return None;
        }
        r.export_attempts = 1;
    }
    let end = checked.iter().fold(
        end.min(row.authority()).min(row.budget.search_end),
        |clock, token| clock.min(token.end),
    );
    let row_count = checked.iter().try_fold(0usize, |n, token| {
        if !token.owned(row) || token.alias_limit > cuts.problem.hidden_dim {
            return None;
        }
        n.checked_add(token.aliases.len())
    })?;
    let mut rows = Vec::with_capacity(row_count);
    let (mut heads, mut terms, mut alias_rows) = (0usize, 0usize, 0usize);
    for token in checked {
        let end = end
            .min(token.end)
            .min(row.authority())
            .min(row.budget.search_end);
        if !token.owned(row) || !live(end) || !poll() {
            return None;
        }
        let (canonical, row_heads) = canonical_row(cuts, parts, binding, token, end, poll)?;
        let copies = token.aliases.len();
        heads = heads.checked_add(row_heads.checked_mul(copies)?)?;
        terms = terms.checked_add(canonical.coeffs.len().checked_mul(copies)?)?;
        if token.alias_limit > 0 {
            let batch = alias::expand(row, cuts, parts, binding, token, &canonical, end, poll)?;
            alias_rows = alias_rows.checked_add(batch.len())?;
            rows.extend(batch);
        } else {
            rows.push(canonical);
        }
    }
    require_hull_environment().ok()?;
    if !live(end) || !poll() {
        return None;
    }
    let mut c = row.budget.state.lock().ok()?;
    let r = &mut c.component_rows.domination;
    r.prepared_rows = rows.len();
    r.prepared_head_terms = heads;
    r.prepared_terms = terms;
    r.prepared_alias_rows = alias_rows;
    r.stop_reason = "prepared";
    Some(rows)
}
