// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-Source K2 allocation; four masks, no nominal residual authority.
use super::super::super::certificate::I;
use super::*;
#[path = "single_hidden_hull_mip_signed_domination_group_kernel_branch.rs"]
pub(in super::super) mod branch;
#[derive(Debug, Clone)]
pub(in super::super) struct Proposal {
    pub negative: [usize; K],
    pub positive: Vec<usize>,
    pub lambda: Vec<f64>,
    pub v: Vec<f64>,
}
pub(in super::super) struct Checked {
    pub(super) proposal: Proposal,
    pub(super) upper: f64,
    pub(super) source: usize,
    pub(super) ledger: Arc<std::sync::Mutex<Counts>>,
    pub(super) witness: Arc<RetainedWords>,
    pub(super) signature: Arc<RetainedWords>,
    pub(super) end: Instant,
    accepted_end: Instant,
    branch_tau: Option<[f64; 4]>,
    support_limit: usize,
    weights: [f64; K],
    bitseal: Vec<u64>,
    export_used: std::sync::atomic::AtomicBool,
}
fn bits(a: &Proposal, weights: &[f64; K], upper: f64) -> Vec<u64> {
    let mut b = Vec::with_capacity(8 + 2 * a.positive.len() + a.v.len());
    b.extend(a.negative.iter().map(|j| *j as u64));
    b.extend([a.positive.len() as u64, a.v.len() as u64]);
    b.extend(a.positive.iter().map(|j| *j as u64));
    b.extend(
        weights
            .iter()
            .chain(&a.lambda)
            .chain(&a.v)
            .map(|x| x.to_bits()),
    );
    b.push(upper.to_bits());
    b
}
fn sealed_bits(a: &Proposal, weights: &[f64; K], upper: f64, tau: Option<[f64; 4]>) -> Vec<u64> {
    let mut seal = bits(a, weights, upper);
    if let Some(tau) = tau {
        seal.push(0x4252414e43485441);
        seal.extend(tau.map(f64::to_bits));
    }
    seal
}

fn shape(p: &HullMarginProblem, a: &Proposal) -> Option<[f64; K]> {
    if !capacity(p)
        || a.negative[0] == a.negative[1]
        || !(1..=32).contains(&a.positive.len())
        || a.positive.len() != a.lambda.len()
        || a.v.len() != p.input_dim
        || a.positive.len().checked_add(a.v.len())?.checked_add(K)? > MAX_TERMS
        || p.w1.len() != p.input_dim.checked_mul(p.hidden_dim)?
        || p.b1.len() != p.hidden_dim
        || p.lower.len() != p.input_dim
        || p.upper.len() != p.input_dim
        || a.v.iter().any(|v| !v.is_finite())
        || a.lambda.iter().any(|v| !v.is_finite() || *v < 0.)
        || !a.lambda.iter().any(|v| *v > 0.)
    {
        return None;
    }
    let weights = [weight(p, a.negative[0])?, weight(p, a.negative[1])?];
    for (i, &j) in a.positive.iter().enumerate() {
        if p.c.get(j).is_none_or(|v| {
            !v.is_finite() || v.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater)
        }) || a.positive[..i].contains(&j)
            || a.negative.contains(&j)
        {
            return None;
        }
    }
    Some(weights)
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
fn add_upper(a: f64, b: f64) -> Option<f64> {
    if a == 0. {
        finite(b)
    } else if b == 0. {
        finite(a)
    } else {
        finite(add_up(a, b))
    }
}
/// lambda and v are the EXACT point coefficients emitted in the final row.
/// B is rebuilt ONCE; r_i=(-Source.c_i)*a_i - B/2 uses outward intervals.
/// The empty mask is essential, as are every bias and all signed coordinates.
pub(in super::super) fn upper(
    p: &HullMarginProblem,
    a: &Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let weights = shape(p, a)?;
    require_hull_environment().ok()?;
    check_clock(end, poll)?;
    let mut b = I::zero();
    for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
        check_clock(end, poll)?;
        b = b.add(I::point(p.b1[j])?.scale(lambda)?)?;
    }
    let biases = [
        I::point(p.b1[a.negative[0]])?
            .scale(weights[0])?
            .sub(b.scale(0.5)?)?,
        I::point(p.b1[a.negative[1]])?
            .scale(weights[1])?
            .sub(b.scale(0.5)?)?,
    ];
    let mut residuals = Vec::with_capacity(p.input_dim);
    for i in 0..p.input_dim {
        check_clock(end, poll)?;
        let mut b = I::zero();
        for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
            b = b.add(I::point(p.w_row(j)[i])?.scale(lambda)?)?;
        }
        let half = b.scale(0.5)?;
        residuals.push([
            I::point(p.w_row(a.negative[0])[i])?
                .scale(weights[0])?
                .sub(half)?,
            I::point(p.w_row(a.negative[1])[i])?
                .scale(weights[1])?
                .sub(half)?,
        ]);
    }
    let mut h = f64::NEG_INFINITY;
    for mask in 0..(1usize << K) {
        check_clock(end, poll)?;
        let mut bias = I::zero();
        for (i, b) in biases.iter().enumerate() {
            if mask & (1 << i) != 0 {
                bias = bias.add(*b)?;
            }
        }
        let mut value = bias.hi;
        for (i, r) in residuals.iter().enumerate() {
            check_clock(end, poll)?;
            let mut coefficient = I::point(-a.v[i])?;
            for (t, c) in r.iter().enumerate() {
                if mask & (1 << t) != 0 {
                    coefficient = coefficient.add(*c)?;
                }
            }
            value = add_upper(value, maximum(coefficient, p.lower[i], p.upper[i])?)?;
        }
        h = h.max(value);
    }
    require_hull_environment().ok()?;
    check_clock(end, poll)?;
    finite(h)
}
pub(in super::super) fn check(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    ticket: &resources::Ticket,
    proposal: Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    let paid = ticket.group_quote()?;
    let end = resources::support_attempt(row, ticket, end)?;
    if !paid.matches(cuts.problem, row.plan.components.len())
        || row.source != std::ptr::from_ref(cuts.problem) as usize
        || row.budget.source != row.source
        || proposal.positive.len() > paid.positive_limit()
    {
        return None;
    }
    let upper = upper(cuts.problem, &proposal, end, poll)?;
    let weights = [
        weight(cuts.problem, proposal.negative[0])?,
        weight(cuts.problem, proposal.negative[1])?,
    ];
    let bitseal = bits(&proposal, &weights, upper);
    {
        let mut c = row.budget.state.lock().ok()?;
        c.component_rows.domination.completed_supports += 1;
        c.component_rows.domination.group_checked_masks += 1usize << K;
    }
    Some(Checked {
        proposal,
        upper,
        weights,
        source: row.source,
        ledger: Arc::clone(&row.budget.state),
        witness: Arc::clone(&row.witness),
        signature: Arc::clone(&row.signature),
        end,
        accepted_end: end,
        branch_tau: None,
        support_limit: paid.positive_limit(),
        bitseal,
        export_used: std::sync::atomic::AtomicBool::new(false),
    })
}
impl Checked {
    pub(super) fn owned(&self, row: &SignedMajorant) -> bool {
        let first = row
            .budget
            .state
            .lock()
            .is_ok_and(|c| c.replay_attempts == 1 && c.source_replays == 1);
        first
            && self.source == row.source
            && self.end == self.accepted_end
            && self.end <= row.authority()
            && self.end <= row.budget.search_end
            && self.upper.is_finite()
            && (1..=32).contains(&self.support_limit)
            && self.proposal.positive.len() <= self.support_limit
            && self.proposal.positive.len() == self.proposal.lambda.len()
            && self.proposal.v.len() == row.budget.dimensions.0
            && Arc::ptr_eq(&self.ledger, &row.budget.state)
            && Arc::ptr_eq(&self.witness, &row.witness)
            && Arc::ptr_eq(&self.signature, &row.signature)
            && self
                .branch_tau
                .is_none_or(|a| a.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)))
            && self.bitseal
                == sealed_bits(&self.proposal, &self.weights, self.upper, self.branch_tau)
    }
}
pub(in super::super) fn export(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    binding: &binding::SignedPhaseBinding<'_>,
    token: &Checked,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    require_hull_environment().ok()?;
    if !binding.matches(cuts, parts) || row.source != std::ptr::from_ref(cuts.problem) as usize {
        return None;
    }
    if token
        .export_used
        .swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        return None;
    }
    {
        let mut c = row.budget.state.lock().ok()?;
        if c.replay_attempts != 1 || c.source_replays != 1 {
            return None;
        }
        let r = &mut c.component_rows.domination;
        if !r.admitted
            || !r.group_allocation
            || token.branch_tau.is_some() != r.branch_tau_mode
            || (token.branch_tau.is_some() && r.branch_tau_checked_masks != 4)
            || r.export_attempts != 0
            || r.completed_supports != 1
            || r.group_checked_masks != 4
        {
            return None;
        }
        r.export_attempts = 1;
    }
    let end = end
        .min(token.end)
        .min(row.authority())
        .min(row.budget.search_end)
        .min(cuts.deadline);
    if !token.owned(row) {
        return None;
    }
    check_clock(end, poll)?;
    let weights = shape(cuts.problem, &token.proposal)?;
    if weights
        .iter()
        .map(|x| x.to_bits())
        .ne(token.weights.iter().map(|x| x.to_bits()))
    {
        return None;
    }
    let a = &token.proposal;
    let mut coeffs = Vec::with_capacity(K + a.positive.len() + a.v.len());
    for (i, &j) in a.negative.iter().enumerate() {
        check_clock(end, poll)?;
        coeffs.push((binding.head_column(j)?.0, weights[i]));
    }
    let mut heads = K;
    for (&j, &lambda) in a.positive.iter().zip(&a.lambda) {
        check_clock(end, poll)?;
        if lambda != 0. {
            coeffs.push((binding.head_column(j)?.0, -lambda));
            heads += 1;
        }
    }
    for (i, &v) in a.v.iter().enumerate() {
        check_clock(end, poll)?;
        if v != 0. {
            coeffs.push((parts.input_vars.get(i)?.0, -v));
        }
    }
    if coeffs.len() > MAX_TERMS {
        return None;
    }
    require_hull_environment().ok()?;
    check_clock(end, poll)?;
    let terms = coeffs.len();
    let rows = vec![RowSpec {
        lb: f64::NEG_INFINITY,
        ub: token.upper,
        coeffs,
    }];
    {
        let mut c = row.budget.state.lock().ok()?;
        if c.replay_attempts != 1 || c.source_replays != 1 {
            return None;
        }
        let r = &mut c.component_rows.domination;
        r.prepared_rows = 1;
        r.prepared_head_terms = heads;
        r.prepared_terms = terms;
        r.stop_reason = "group-prepared";
    }
    Some(rows)
}
