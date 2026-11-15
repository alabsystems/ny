// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optional row-dual search after an already complete source-owned cover.
//! Exported rows are advice snapshots. Only the phase multipliers return;
//! source intervals, bias charges and full Q residuals remain locally owned.

use std::sync::{Arc, OnceLock};

use ny_core::head_phase_advice::{
    HeadPhaseAdvice, HeadPhaseLpRequest, HeadPhaseLpRow, PROTOCOL_VERSION,
};

use super::*;

const MAX_CALLS: usize = 2;
const TOTAL_PROPOSAL_TIME: Duration = Duration::from_millis(250);
const PER_CALL_TIME: Duration = Duration::from_millis(125);
const REPLAY_RESERVE: Duration = Duration::from_millis(50);

type AdviceFactory = Box<dyn Fn() -> Option<Arc<dyn HeadPhaseAdvice>> + Send + Sync>;
static FACTORY: OnceLock<AdviceFactory> = OnceLock::new();

/// Install the optional head-phase advice factory. First registration wins.
/// The closure is invoked lazily after a complete owned native cover exists;
/// startup registration performs no AY initialization or helper spawning.
pub fn register_head_phase_advice_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

pub(super) fn registered_provider() -> Option<Arc<dyn HeadPhaseAdvice>> {
    static HANDLE: OnceLock<Option<Arc<dyn HeadPhaseAdvice>>> = OnceLock::new();
    let factory = FACTORY.get()?;
    HANDLE
        .get_or_init(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(factory))
                .ok()
                .flatten()
        })
        .as_ref()
        .map(Arc::clone)
}

pub(super) fn has_factory() -> bool {
    FACTORY.get().is_some()
}

/// One invocation's allowance, shared across classes, groups and masks.
pub(super) struct AdviceBudget {
    pub(super) calls: usize,
    pub(super) charged: Duration,
}
impl AdviceBudget {
    pub(super) fn new() -> Self {
        Self {
            calls: 0,
            charged: Duration::ZERO,
        }
    }
    pub(super) fn charge_elapsed(&mut self, elapsed: Duration) {
        self.charged = self.charged.saturating_add(elapsed);
    }
    /// Native replay consumes the SAME total advice allowance. No new call
    /// slot is required; no extra wall-time allowance is created.
    pub(super) fn replay_deadline(&self, now: Instant, deadline: Instant) -> Option<Instant> {
        let remaining = TOTAL_PROPOSAL_TIME.checked_sub(self.charged)?;
        (!remaining.is_zero())
            .then(|| now.checked_add(remaining))
            .flatten()
            .map(|end| end.min(deadline))
            .filter(|&end| end > now)
    }
    pub(super) fn call_deadline(&self, now: Instant, deadline: Instant) -> Option<Instant> {
        if self.calls >= MAX_CALLS || self.charged >= TOTAL_PROPOSAL_TIME {
            return None;
        }
        let available = deadline
            .checked_duration_since(now)?
            .checked_sub(REPLAY_RESERVE)?;
        let allowance = available
            .min(PER_CALL_TIME)
            .min(TOTAL_PROPOSAL_TIME.checked_sub(self.charged)?);
        (!allowance.is_zero())
            .then(|| now.checked_add(allowance))
            .flatten()
    }
}

#[derive(Default)]
pub(super) struct AdviceRefinement {
    pub calls: usize,
    pub candidates: usize,
    pub improved_phases: usize,
    pub native_passes: usize,
    pub native_columns: usize,
}

fn midpoint(v: I) -> Option<f64> {
    let point = 0.5 * v.lo + 0.5 * v.hi;
    good(point).then_some(point)
}

/// Export source-owned point coefficients and repaired bias endpoints. The
/// nominal objective guides search; its constant and interval discrepancies
/// remain in NY's owned replay and are never returned by the provider.
fn export_phase_lp(
    problem: &SourceHeadProblem,
    group: &[usize],
    rest: &Affine,
    mask: usize,
    deadline: Instant,
) -> Option<HeadPhaseLpRequest> {
    problem.validate()?;
    if !live(deadline)
        || !rest.valid(problem.root.len())
        || !(2..=MAX_GROUP).contains(&group.len())
        || mask >= 1 << group.len()
        || group
            .iter()
            .enumerate()
            .any(|(i, &j)| j >= problem.head.len() || group[..i].contains(&j))
    {
        return None;
    }
    let n = problem.root.len();
    let width = n.checked_add(group.len())?;
    let mut request = HeadPhaseLpRequest {
        protocol_version: PROTOCOL_VERSION,
        bounds: problem.root.iter().map(|x| [x.lo, x.hi]).collect(),
        objective: Vec::with_capacity(width),
        rows: Vec::with_capacity(3 * group.len()),
        phase_rows: Vec::with_capacity(group.len()),
    };
    for (j, &a) in rest.a.iter().enumerate() {
        if j & 63 == 0 && !live(deadline) {
            return None;
        }
        request.objective.push(midpoint(a)?);
    }
    for (i, &j) in group.iter().enumerate() {
        let e = &problem.head[j];
        let active = mask & (1 << i) != 0;
        let eta = if active { 1.0 } else { -1.0 };
        request.bounds.push([e.y.lo, e.y.hi]);
        request
            .objective
            .push(if active { midpoint(problem.q[j])? } else { 0.0 });
        let mut lower = vec![0.0; width];
        let mut upper = vec![0.0; width];
        let mut phase = vec![0.0; width];
        for k in 0..n {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            // owned_row stores exact binary64 coefficient points. An interval
            // row needs a different exporter and is deliberately declined.
            let (a, b) = (e.lower.a[k], e.upper.a[k]);
            if a.lo != a.hi || b.lo != b.hi {
                return None;
            }
            lower[k] = -a.lo;
            upper[k] = b.lo;
        }
        lower[n + i] = 1.0;
        upper[n + i] = -1.0;
        phase[n + i] = eta;
        request.rows.push(HeadPhaseLpRow {
            coefficients: lower,
            lower: e.lower.b.lo,
        });
        request.rows.push(HeadPhaseLpRow {
            coefficients: upper,
            lower: -e.upper.b.hi,
        });
        request.phase_rows.push(request.rows.len());
        request.rows.push(HeadPhaseLpRow {
            coefficients: phase,
            lower: 0.0,
        });
    }
    (live(deadline) && request.validate()).then_some(request)
}

// Transfer a rounded nominal functional anchor across CLOSED masks. These are
// merely new finite nonnegative multiplier proposals: every mask is separately
// replayed, with the complete authored Q interval and its full Y residual.
fn transfer_phase_anchor(
    problem: &SourceHeadProblem,
    group: &[usize],
    origin: &Proposal,
    mask: usize,
) -> Option<Proposal> {
    if origin.mask >= 1 << group.len()
        || mask >= 1 << group.len()
        || origin.phase.len() != group.len()
    {
        return None;
    }
    let mut phase = Vec::with_capacity(group.len());
    for (i, &j) in group.iter().enumerate() {
        let q = midpoint(problem.q[j])?;
        let old_s = f64::from(u8::from(origin.mask & (1 << i) != 0));
        let s = f64::from(u8::from(mask & (1 << i) != 0));
        let mu = q * old_s - (2.0 * old_s - 1.0) * origin.phase[i];
        let p = (2.0 * s - 1.0) * (q * s - mu);
        if !good(mu) || !good(p) {
            return None;
        }
        phase.push(p.max(0.0));
    }
    Some(Proposal { mask, phase })
}

/// Retain all prior mask certificates; submitted signs only offer candidates.
/// No LP status, optimum, feasibility or returned model is read here.
pub(super) fn refine_complete_cover(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    cover: &mut RootGroupBound<'_>,
    budget: &mut AdviceBudget,
    deadline: Instant,
) -> Option<AdviceRefinement> {
    if FACTORY.get().is_none() {
        return Some(AdviceRefinement::default());
    }
    refine_with_provider(
        eng,
        pack,
        problem,
        group,
        cover,
        budget,
        deadline,
        registered_provider,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn refine_with_provider<F>(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    cover: &mut RootGroupBound<'_>,
    budget: &mut AdviceBudget,
    deadline: Instant,
    provider: F,
) -> Option<AdviceRefinement>
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let mut report = AdviceRefinement::default();
    if !cover.applies_to(problem)
        || cover.group != group
        || cover.phases != 1 << group.len()
        || cover.phase_lower.len() != cover.phases
        || cover.proposals.len() != cover.phases
        || cover.phase_lower.iter().any(|v| !good(*v))
        || !fp_qualified()
        || !pack.owned_for_head_groups(eng)
    {
        return None;
    }
    if !live(deadline) || cover.lower > 0.0 {
        return Some(report);
    }
    let preparation_start = Instant::now();
    let Some(preparation_deadline) = budget.call_deadline(preparation_start, deadline) else {
        return Some(report);
    };
    let rest = remainder(problem, group, preparation_deadline);
    budget.charged = budget.charged.saturating_add(preparation_start.elapsed());
    let Some(rest) = rest.filter(|_| live(preparation_deadline)) else {
        return Some(report);
    };
    let before = cover.phase_lower.clone();
    let mut masks: Vec<_> = (0..cover.phases).collect();
    masks.sort_by(|&a, &b| {
        cover.phase_lower[a]
            .total_cmp(&cover.phase_lower[b])
            .then(a.cmp(&b))
    });
    for mask in masks {
        if cover.phase_lower[mask] > 0.0 {
            continue;
        }
        let begin = Instant::now();
        let Some(call_deadline) = budget.call_deadline(begin, deadline) else {
            break;
        };
        budget.calls += 1;
        // Include exporter, lazy provider construction, helper setup/solve and
        // receipt validation in the same proposal allowance. NY replay uses
        // the remaining local/original deadline, with an explicit reserve.
        let Some(request) = export_phase_lp(problem, group, &rest, mask, call_deadline) else {
            budget.charge_elapsed(begin.elapsed());
            break;
        };
        let backend = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&provider))
            .ok()
            .flatten();
        let Some(backend) = backend else {
            budget.charged = budget.charged.saturating_add(begin.elapsed());
            break;
        };
        let hint = if live(call_deadline) {
            report.calls += 1;
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.propose(&request, call_deadline)
            }))
            .ok()
            .flatten()
        } else {
            None
        };
        let timely = live(call_deadline);
        let hint = hint.filter(|h| h.validate(&request));
        budget.charged = budget.charged.saturating_add(begin.elapsed());
        if !fp_qualified() || !pack.owned_for_head_groups(eng) {
            return None;
        }
        let Some(hint) = hint.filter(|_| timely && live(call_deadline)) else {
            continue;
        };
        let mut best_native: Vec<Option<(f64, Proposal)>> = vec![None; cover.phases];
        let mut origins = Vec::with_capacity(2);
        for sign in [1.0, -1.0] {
            let phase: Vec<_> = request
                .phase_rows
                .iter()
                .map(|&r| (sign * hint.row_duals[r]).max(0.0))
                .collect();
            if origins.iter().any(|p: &Proposal| p.phase == phase) {
                continue;
            }
            origins.push(Proposal { mask, phase });
        }
        for origin in origins {
            for other in 0..cover.phases {
                if !live(deadline) {
                    break;
                }
                let Some(proposal) = transfer_phase_anchor(problem, group, &origin, other) else {
                    continue;
                };
                if let Some(lower) = phase_lower(problem, group, &rest, &proposal, deadline) {
                    if lower > cover.phase_lower[other] {
                        cover.phase_lower[other] = lower;
                        cover.proposals[other] = proposal.clone();
                    }
                    if best_native[other]
                        .as_ref()
                        .is_none_or(|(old, _)| lower > *old)
                    {
                        best_native[other] = Some((lower, proposal));
                    }
                    report.candidates += 1;
                }
            }
        }
        cover.lower = cover
            .phase_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if cover.lower > 0.0 {
            break;
        }
        let mut weak: Vec<_> = (0..cover.phases)
            .filter(|&m| cover.phase_lower[m] <= 0.0)
            .collect();
        weak.sort_by(|&a, &b| {
            cover.phase_lower[a]
                .total_cmp(&cover.phase_lower[b])
                .then(a.cmp(&b))
        });
        // Breadth first: one retained candidate per weak mask before spending
        // native columns on alternatives for an already represented mask.
        let proposals: Vec<_> = weak
            .into_iter()
            .filter_map(|m| best_native[m].take().map(|(_, p)| p))
            .take(MAX_NATIVE_COLUMNS)
            .collect();
        // Both sign conventions may also improve the stronger whole-functional
        // native path. Every complete column retains its full authored Q and
        // scalar remainder residual; incomplete columns cannot replace a mask.
        if let Some(replayed) =
            native::replay_proposals(eng, pack, problem, group, &proposals, deadline)
        {
            report.native_passes += replayed.passes;
            report.native_columns += replayed.columns;
            for (index, lower) in replayed.bounds {
                let other = proposals[index].mask;
                if lower > cover.phase_lower[other] {
                    cover.phase_lower[other] = lower;
                    cover.proposals[other] = proposals[index].clone();
                }
            }
        }
        if !fp_qualified() || !pack.owned_for_head_groups(eng) {
            return None;
        }
        cover.lower = cover
            .phase_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if !live(deadline) || cover.lower > 0.0 {
            break;
        }
    }
    report.improved_phases = before
        .iter()
        .zip(&cover.phase_lower)
        .filter(|(a, b)| b > a)
        .count();
    (fp_qualified() && cover.phase_lower.iter().all(|v| good(*v))).then_some(report)
}

#[cfg(test)]
#[path = "head_group_advice_tests.rs"]
mod tests;
