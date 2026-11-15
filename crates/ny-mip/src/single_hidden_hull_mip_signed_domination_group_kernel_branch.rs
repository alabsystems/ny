// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Four ORIGINAL-Source affine supports with independent point tau in [0,1].
use super::super::super::branch as nominal;
use super::*;
/// No nominal residual or breakpoint value is a bound. Rebuild every bias and
/// coordinate of A0,A1,B using directed intervals and the exact emitted lambda.
pub(in super::super::super) fn upper(
    p: &HullMarginProblem,
    a: &Proposal,
    tau: [f64; 4],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let weights = shape(p, a)?;
    if tau
        .iter()
        .any(|t| !t.is_finite() || !(0.0..=1.0).contains(t))
    {
        return None;
    }
    require_hull_environment().ok()?;
    check_clock(end, poll)?;
    let mut b = I::zero();
    for (&j, &l) in a.positive.iter().zip(&a.lambda) {
        check_clock(end, poll)?;
        b = b.add(I::point(p.b1[j])?.scale(l)?)?;
    }
    let biases = [
        I::point(p.b1[a.negative[0]])?.scale(weights[0])?,
        I::point(p.b1[a.negative[1]])?.scale(weights[1])?,
    ];
    let mut coordinates = Vec::with_capacity(p.input_dim);
    for i in 0..p.input_dim {
        check_clock(end, poll)?;
        let mut sum = I::zero();
        for (&j, &l) in a.positive.iter().zip(&a.lambda) {
            check_clock(end, poll)?;
            sum = sum.add(I::point(p.w_row(j)[i])?.scale(l)?)?;
        }
        coordinates.push((
            [
                I::point(p.w_row(a.negative[0])[i])?.scale(weights[0])?,
                I::point(p.w_row(a.negative[1])[i])?.scale(weights[1])?,
            ],
            sum,
        ));
    }
    let mut upper = f64::NEG_INFINITY;
    for mask in 0..4 {
        check_clock(end, poll)?;
        let mut bias = I::zero().sub(b.scale(tau[mask])?)?;
        for t in 0..2 {
            if mask & (1 << t) != 0 {
                bias = bias.add(biases[t])?;
            }
        }
        let mut value = bias.hi;
        for (i, (aa, bb)) in coordinates.iter().enumerate() {
            check_clock(end, poll)?;
            let mut coefficient = I::zero().sub(bb.scale(tau[mask])?)?;
            for t in 0..2 {
                if mask & (1 << t) != 0 {
                    coefficient = coefficient.add(aa[t])?;
                }
            }
            coefficient = coefficient.sub(I::point(a.v[i])?)?;
            value = add_upper(value, maximum(coefficient, p.lower[i], p.upper[i])?)?;
        }
        upper = upper.max(value);
    }
    require_hull_environment().ok()?;
    check_clock(end, poll)?;
    finite(upper)
}
pub(in super::super::super) fn check(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    ticket: &resources::Ticket,
    proposal: nominal::Proposal,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    let paid = ticket.branch_quote()?;
    let end = resources::branch_support_attempt(row, ticket, end)?.min(cuts.deadline);
    if !paid.matches(cuts.problem, row.plan.components.len())
        || row.source != std::ptr::from_ref(cuts.problem) as usize
        || row.budget.source != row.source
        || proposal.affine.positive.len() > paid.base().positive_limit
    {
        return None;
    }
    let upper = upper(cuts.problem, &proposal.affine, proposal.tau, end, poll)?;
    // upper already authenticated the full immutable proposal. Avoid a third
    // quadratic duplicate pass; the quote pays shape at support and export.
    let weights = [
        weight(cuts.problem, proposal.affine.negative[0])?,
        weight(cuts.problem, proposal.affine.negative[1])?,
    ];
    let branch_tau = Some(proposal.tau);
    let bitseal = sealed_bits(&proposal.affine, &weights, upper, branch_tau);
    {
        let mut c = row.budget.state.lock().ok()?;
        let r = &mut c.component_rows.domination;
        r.completed_supports += 1;
        r.group_checked_masks += 4;
        r.branch_tau_checked_masks += 4;
    }
    Some(Checked {
        proposal: proposal.affine,
        upper,
        source: row.source,
        ledger: Arc::clone(&row.budget.state),
        witness: Arc::clone(&row.witness),
        signature: Arc::clone(&row.signature),
        end,
        accepted_end: end,
        branch_tau,
        support_limit: paid.base().positive_limit,
        weights,
        bitseal,
        export_used: std::sync::atomic::AtomicBool::new(false),
    })
}
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_group_kernel_branch_tests.rs"]
mod tests;
