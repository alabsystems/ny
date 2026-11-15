// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One immutable paid dense fit; no row/checker authority is minted.
use super::super::super::super::primal_advice::Point;
use super::*;
#[path = "single_hidden_hull_mip_signed_domination_dense_fused.rs"]
mod fused;
const CAP: usize = 664_696;
const MAX_TERMS: usize = 64;
#[derive(Clone)]
pub(super) struct Quote {
    pub total_work: usize,
    pub incremental_work: usize,
    pub bytes: usize,
    pub positive_limit: usize,
    pub pool_heads: usize,
    pub negative_heads: usize,
    pub alias_work: usize,
    pub alias_bytes: usize,
    pub step_work: usize,
    pub step_bytes: usize,
    pub sort_work: usize,
}
impl Quote {
    /// The captured debt and retained incremental quote must describe one debit.
    pub(super) fn checked_total(&self, capture: usize) -> Option<usize> {
        let total = capture.checked_add(self.incremental_work)?;
        (total == self.total_work && total <= CAP).then_some(total)
    }
}
/// Automatic dense admission includes the full captured-negative equality/copy
/// batch, before inspecting which original affines are equal.
pub(super) fn quote_with_aliases(
    p: &HullMarginProblem,
    components: usize,
    capture_work: usize,
    pool_heads: usize,
    negative_heads: usize,
    positive_limit: usize,
) -> Option<Quote> {
    let (d, h) = (p.input_dim, p.hidden_dim);
    let q = positive_limit
        .min(pool_heads)
        .min(32)
        .min(63usize.checked_sub(d)?);
    // The original planner may retain one component per hidden head, including
    // stable heads with a downward-rounded negative margin coefficient. The
    // quadratic metadata charge below, together with CAP, bounds that work.
    if q == 0
        || d == 0
        || d > 128
        || h > 512
        || components > h
        || pool_heads > h
        || negative_heads > h
    {
        return None;
    }
    let width = d.checked_add(1)?;
    let targets = h
        .checked_mul(4)?
        .checked_mul(16)?
        .checked_add(width.checked_mul(4)?.checked_mul(64)?)?;
    // Immutable midpoint/halfwidth/target cache, then one positive alignment.
    // Pool14FP/9finite ->7FP/6finite; fit16FP/10finite ->6FP/5finite
    // per coordinate, with unchanged2FP/2finite residual updates.
    let features = width
        .checked_mul(64)?
        .checked_add(pool_heads.checked_mul(width)?.checked_mul(40)?)?;
    // Stable Feature24P records; ranked16q tuples pay indirect head lookups.
    let ranking = pool_heads.checked_mul(q)?.checked_mul(24)?;
    let fit = q.checked_mul(width)?.checked_mul(64)?;
    let scalar = width.checked_mul(256)?;
    // The unchanged old oriented comparator uses an index merge sort.
    // The new fixed-t engine sorts separately. BOTH are explicitly prepaid.
    let levels = if d <= 1 {
        0
    } else {
        usize::BITS as usize - (d - 1).leading_zeros() as usize
    };
    let sort_work = d.checked_mul(levels)?.checked_mul(192)?;
    let chain = sort_work.checked_add(width.checked_mul(256)?)?;
    let step_work = q
        .checked_mul(width)?
        .checked_mul(128)?
        .checked_add(width.checked_mul(1024)?)?
        .checked_add(4096)?;
    let step_bytes = d
        .checked_mul(96)?
        .checked_add(q.checked_mul(64)?)?
        .checked_add(2048)?;
    // Extra stable Features + three coordinate caches + old-chain index buffers.
    let cache_bytes = d
        .checked_mul(40)?
        .checked_add(pool_heads.checked_mul(24)?)?
        .checked_add(256)?;
    // shape checks duplicates both before support and before export:
    // at most q(q-1) comparisons total, paid separately at8units each.
    let duplicate_checks = q.checked_mul(q.checked_sub(1)?)?.checked_mul(8)?;
    let checker = q
        .checked_add(1)?
        .checked_mul(width)?
        .checked_mul(128)?
        .checked_add(duplicate_checks)?;
    let export = d.checked_add(q)?.checked_add(1)?.checked_mul(128)?;
    let metadata = components
        .checked_mul(components)?
        .checked_mul(32)?
        .checked_add(components.checked_mul(128)?)?
        .checked_add(8192)?;
    let (alias_work, alias_bytes) = alias::quote(d, q, negative_heads)?;
    let incremental_work = targets
        .checked_add(features)?
        .checked_add(ranking)?
        .checked_add(fit)?
        .checked_add(scalar)?
        .checked_add(chain)?
        .checked_add(checker)?
        .checked_add(export)?
        .checked_add(metadata)?
        .checked_add(alias_work)?
        .checked_add(step_work)?;
    let total_work = capture_work.checked_add(incremental_work)?;
    // Simultaneous pool, residual, scalar/chain directions, point witness,
    // original-support bit seal, and exported row. Nominal Vec payload only.
    let bytes = h
        .checked_add(q.checked_mul(8)?)?
        .checked_add(width.checked_mul(16)?)?
        .checked_add(MAX_TERMS.checked_mul(4)?)?
        .checked_mul(64)?
        .checked_add(8192)?
        .checked_add(alias_bytes)?
        .checked_add(step_bytes)?
        .checked_add(cache_bytes)?;
    (total_work <= CAP).then_some(Quote {
        total_work,
        incremental_work,
        bytes,
        positive_limit: q,
        pool_heads,
        negative_heads,
        alias_work,
        alias_bytes,
        step_work,
        step_bytes,
        sort_work,
    })
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn center(l: f64, u: f64) -> Option<f64> {
    finite(0.5 * l + 0.5 * u)
}
pub(super) fn direction(c: &[f64], bias: f64, p: &HullMarginProblem) -> Option<Vec<f64>> {
    let (mut lo, mut hi) = (bias, bias);
    for (i, &w) in c.iter().enumerate() {
        let a = finite(w * p.lower[i])?;
        let b = finite(w * p.upper[i])?;
        lo = finite(lo + a.min(b))?;
        hi = finite(hi + a.max(b))?;
    }
    let rho = if lo >= 0. {
        1.
    } else if hi <= 0. {
        0.
    } else {
        finite(hi / finite(hi - lo)?)?.clamp(0., 1.)
    };
    c.iter().map(|w| finite(rho * w)).collect()
}
/// Plain-float price of the SAME two affine branches as the real checker.
/// It selects a direction only; the checked constant is never taken here.
pub(super) fn price(
    c: &[f64],
    bias: f64,
    v: &[f64],
    p: &HullMarginProblem,
    x: &[f64],
) -> Option<f64> {
    let (mut zero, mut active, mut at) = (0., bias, 0.);
    for i in 0..p.input_dim {
        let z = -v[i];
        let r = finite(c[i] - v[i])?;
        zero = finite(zero + finite(z * p.lower[i])?.max(finite(z * p.upper[i])?))?;
        active = finite(active + finite(r * p.lower[i])?.max(finite(r * p.upper[i])?))?;
        at = finite(at + finite(v[i] * x[i])?)?;
    }
    finite(zero.max(active) + at)
}
/// A lower additive price alone cannot replace an old subtractive violation:
/// rounded accumulation can disagree. Keep the old row unless the replacement
/// also violates and retains at least one genuinely positive partner weight.
fn accept_priced(candidate: &priced::Candidate, old: Option<f64>, target: f64) -> bool {
    old.is_some_and(|value| candidate.value < value)
        && target > candidate.value
        && candidate.lambda.iter().any(|a| *a > 0.)
}
pub(super) struct Advice {
    pub single: kernel::Proposal,
    pub useful: bool,
    pub partner: Option<usize>,
}
/// Must be called once ONLY after the dense Ticket paid quote.incremental_work,
/// exact payload, and one original checker slot. No proof or spend occurs here.
pub(super) fn advice(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    point: &Point,
    paid: &Quote,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Advice> {
    advice_policy(row, cuts, point, paid, None, end, poll)
}
pub(super) fn exclusive_advice(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    point: &Point,
    paid: &Quote,
    ticket: &resources::Ticket,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Advice> {
    advice_policy(row, cuts, point, paid, Some(ticket), end, poll)
}
fn advice_policy(
    row: &SignedMajorant,
    cuts: &HullRootMipCuts<'_>,
    point: &Point,
    paid: &Quote,
    ticket: Option<&resources::Ticket>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Advice> {
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
    if let Ok(mut state) = row.budget.state.lock() {
        let r = &mut state.component_rows.domination;
        r.priced_step_work_units = paid.step_work;
        r.priced_sort_work_units = paid.sort_work;
        r.priced_step_bytes = paid.step_bytes;
    }
    let p = cuts.problem;
    let end = end
        .min(row.authority())
        .min(row.budget.search_end)
        .min(cuts.deadline);
    let d = p.input_dim;
    let q = paid.positive_limit.min(32).min(63usize.checked_sub(d)?);
    if q == 0
        || d == 0
        || point.input.len() != d
        || point.values.len() != p.hidden_dim
        || point.positive.len() > paid.pool_heads
        || point.negative.len() > p.hidden_dim
        || !live(end)
        || !poll()
    {
        return None;
    }
    let mut targets: Vec<(f64, usize)> = Vec::with_capacity(4);
    for &j in &point.negative {
        if !live(end) || !poll() {
            return None;
        }
        let c = *p.c.get(j)?;
        let y = finite(*point.values.get(j)?)?;
        let score = finite((-c).max(0.) * y.max(0.))?;
        if c < 0. && score > 0. && !targets.iter().any(|(_, old)| *old == j) {
            let slot = targets
                .iter()
                .position(|&(old, k)| score > old || (score == old && j < k))
                .unwrap_or(targets.len());
            if slot < 4 {
                targets.insert(slot, (score, j));
                if targets.len() > 4 {
                    targets.pop();
                }
            }
        }
    }
    let mut gaps: Vec<(f64, usize)> = Vec::with_capacity(4);
    let mut target = None;
    for (_, j) in targets {
        let mut affine = p.b1[j];
        for (i, &x) in point.input.iter().enumerate() {
            if !live(end) || !poll() {
                return None;
            }
            affine = finite(affine + finite(p.w_row(j)[i] * x)?)?;
        }
        let gap = finite((-p.c[j]) * finite(point.values[j] - affine.max(0.))?.max(0.))?;
        if gap > 0. {
            gaps.push((gap, j));
        }
        if gap > 0.
            && target.is_none_or(|(old, k): (f64, usize)| gap > old || (gap == old && j < k))
        {
            target = Some((gap, j));
        }
    }
    let head = target?.1;
    let partner = gaps
        .into_iter()
        .filter(|(_, j)| *j != head)
        .max_by(|(a, i), (b, j)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .map(|(_, j)| j);
    let seed = fused::seed(p, point, head, q, end, poll)?;
    let fused::Seed {
        positive,
        mut lambda,
        residual,
        bias,
    } = seed;
    if positive.is_empty() {
        return None;
    }
    let mut v = direction(&residual, bias, p)?;
    let mut best = price(&residual, bias, &v, p, &point.input)?;
    if !live(end) || !poll() {
        return None;
    }
    // Known single-ReLU closure applied to the synthesized mixed-head residual.
    // A failed or worse nominal chain leaves the original scalar choice intact.
    if let Some(chain) =
        chain::nominal_chain_direction(&residual, bias, &p.lower, &p.upper, &point.input)
    {
        if let Some(value) = price(&residual, bias, &chain, p, &point.input) {
            if value < best {
                best = value;
                v = chain;
            }
        }
    }
    if !live(end) || !poll() {
        return None;
    }
    // Preserve the old finite seed, scalar/old-oriented-chain selection and
    // subtractive violation test for this SAME funded q. Zero partners never
    // change the old accumulation order or signed-zero arithmetic.
    let mut lhs = finite(point.values[head])?;
    let mut head_price = Some(0.);
    for (&j, &a) in positive.iter().zip(&lambda) {
        if a == 0. {
            continue;
        }
        lhs = finite(lhs - finite(a * point.values[j])?)?;
        head_price = head_price.and_then(|old| finite(old + finite(a * point.values[j])?));
    }
    let mut useful = lhs > best;
    let full_seed_price = head_price.and_then(|sum| finite(sum + best));
    // Route selection precedes BOTH optional tails and is irreversible. A
    // known-impossible pair retains the old gradient opportunity.
    let run_gradient = if let Some(ticket) = ticket {
        // The quoted64q serialization allowance also pays this eligibility
        // pass: do not route a tiny anchor to a known all-zero scaled vector.
        let scale = -p.c[head];
        let serializable = lambda.iter().try_fold(false, |positive, a| {
            finite(scale * a).map(|value| positive || value > 0.)
        });
        let pair = partner.is_some_and(|j| j != head)
            && scale.is_finite()
            && scale > 0.
            && serializable == Some(true);
        resources::select_branch_route(row, ticket, useful, pair)?
    } else {
        true
    };
    if run_gradient {
        let partners: Vec<_> = positive
            .iter()
            .map(|&j| priced::Head {
                weights: p.w_row(j),
                bias: p.b1[j],
                value: point.values[j],
            })
            .collect();
        let target = priced::Head {
            weights: p.w_row(head),
            bias: p.b1[head],
            value: point.values[head],
        };
        // A second, separately paid stable fixed-t sort, complete means (including
        // zero residual coordinates), and at most ONE projected coefficient step.
        if let Some(candidate) = priced::improve(
            target,
            &partners,
            &lambda,
            &p.lower,
            &p.upper,
            &point.input,
            end,
            poll,
        ) {
            let selected = accept_priced(&candidate, full_seed_price, point.values[head]);
            if let Ok(mut state) = row.budget.state.lock() {
                let r = &mut state.component_rows.domination;
                r.priced_step_attempted = candidate.attempted;
                r.priced_step_completed = candidate.completed;
                r.priced_step_accepted = candidate.improved && selected;
            }
            if selected {
                useful = point.values[head] > candidate.value;
                lambda = candidate.lambda;
                v = candidate.slope;
            }
        }
    }
    if !live(end) || !poll() || positive.len().checked_add(d)?.checked_add(1)? > MAX_TERMS {
        return None;
    }
    // An unchanged seed has exactly the original nonzero partner sequence.
    let (positive, lambda) = positive
        .into_iter()
        .zip(lambda)
        .filter(|(_, a)| *a != 0.)
        .unzip();
    Some(Advice {
        single: kernel::Proposal {
            head,
            positive,
            lambda,
            v,
        },
        useful,
        partner,
    })
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_domination_dense_tests.rs"]
mod tests;
