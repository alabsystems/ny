// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Global scalar slopes are search proposals. The projected LP has no source
//! authority; native replay retains all heads, features and full Q/Y charges.

use super::advice::{AdviceBudget, AdviceRefinement};
use super::*;
use ny_core::head_functional_advice::{
    HeadFunctionalLpHint, HeadFunctionalLpRequest, HeadFunctionalLpRow, FUNCTIONAL_MAX_FEATURES,
    FUNCTIONAL_PROTOCOL_VERSION,
};
use ny_core::head_phase_advice::HeadPhaseAdvice;
use std::sync::Arc;

#[derive(Default)]
#[cfg(test)]
pub(super) struct UnconditionalRefinement {
    pub lower: Option<f64>,
    pub report: AdviceRefinement,
}

fn baseline_slopes(
    problem: &SourceHeadProblem,
    group: &[usize],
    deadline: Instant,
) -> Option<Vec<f64>> {
    problem
        .head
        .iter()
        .enumerate()
        .map(|(j, e)| {
            if !live(deadline) {
                return None;
            }
            if group.contains(&j) {
                Some(0.0)
            } else {
                scalar_line(problem.q[j], e).map(|(v, _)| v)
            }
        })
        .collect()
}

/// A complete unconditional scalar-box certificate, with no affine-row scan.
/// These Q/Y intervals are the private producer's full owned intervals.
fn unconditional_baseline(problem: &SourceHeadProblem, deadline: Instant) -> Option<f64> {
    if problem.q.len() != problem.head.len() || !problem.bias.valid() {
        return None;
    }
    let mut bound = problem.bias;
    for (q, e) in problem.q.iter().zip(&problem.head) {
        if !live(deadline) || !e.y.valid() {
            return None;
        }
        bound = bound.add(q.mul(I {
            lo: e.y.lo.max(0.0),
            hi: e.y.hi.max(0.0),
        })?)?;
    }
    (live(deadline) && good(bound.lo)).then_some(bound.lo)
}

/// Only search geometry: sampled support is expanded and each row's nominal
/// center is calibrated to its owned Y midpoint. No omitted feature scan, box
/// or resulting LP row replaces a private source row.
fn export_functional_lp(
    eng: &BackwardEngine<'_>,
    problem: &SourceHeadProblem,
    group: &[usize],
    mask: usize,
    deadline: Instant,
) -> Option<HeadFunctionalLpRequest> {
    if !live(deadline)
        || problem.head.is_empty()
        || problem.head.len() > MAX_HEAD
        || problem.q.len() != problem.head.len()
        || problem.q.iter().any(|q| !q.valid())
        || problem.head.iter().any(|e| !e.y.valid())
        || group.len() > MAX_GROUP
        || mask >= 1 << group.len()
        || group
            .iter()
            .enumerate()
            .any(|(i, &j)| j >= problem.head.len() || group[..i].contains(&j))
    {
        return None;
    }
    let (weights, biases, (heads, width)) = eng.net.gemm1();
    if heads != problem.head.len()
        || width == 0
        || weights.len() != heads.checked_mul(width)?
        || biases.len() != heads
    {
        return None;
    }
    let samples = width.min(FUNCTIONAL_MAX_FEATURES);
    let features: Vec<_> = (0..samples)
        .map(|k| {
            if samples == 1 {
                Some(0)
            } else {
                k.checked_mul(width - 1).map(|v| v / (samples - 1))
            }
        })
        .collect::<Option<_>>()?;
    let mut tensor = match &eng.net.ops[eng.net.i_gemm1] {
        TwinOp::Gemm { input, .. } => *input,
        _ => return None,
    };
    while tensor > 0 {
        if !live(deadline) {
            return None;
        }
        match &eng.net.ops[tensor - 1] {
            TwinOp::Flatten { input } => tensor = *input,
            _ => break,
        }
    }
    let last = tensor.checked_sub(1).and_then(|op| {
        eng.root
            .layers
            .iter()
            .find(|layer| layer.op == op && layer.n == width)
    });
    let geometry = |k: usize| -> I {
        if let Some(layer) = last {
            I {
                lo: layer.l[k].max(0.0),
                hi: layer.u[k].max(0.0),
            }
        } else if tensor == 0 && width == problem.root.len() {
            problem.root[k]
        } else {
            I { lo: -1.0, hi: 1.0 }
        }
    };
    let sample_scale = width as f64 / samples as f64;
    let centers: Vec<_> = features
        .iter()
        .map(|&k| {
            let x = geometry(k);
            0.5 * x.lo + 0.5 * x.hi
        })
        .collect();
    let boxes: Vec<_> = features
        .iter()
        .zip(&centers)
        .map(|(&k, &center)| {
            let x = geometry(k);
            let radius = (0.5 * x.hi - 0.5 * x.lo) * sample_scale;
            I {
                lo: center - radius,
                hi: center + radius,
            }
        })
        .collect();
    if boxes.iter().any(|x| !x.valid()) {
        return None;
    }
    let mut request = HeadFunctionalLpRequest {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        feature_count: samples,
        bounds: boxes.iter().map(|x| [x.lo, x.hi]).collect(),
        objective: vec![0.0; samples],
        rows: Vec::new(),
        slope_rows: vec![None; heads],
        phase_rows: Vec::with_capacity(group.len()),
    };
    let baseline = baseline_slopes(problem, group, deadline)?;
    let mut projected = Vec::with_capacity(heads);
    let mut projected_bias = Vec::with_capacity(heads);
    for j in 0..heads {
        let mut b = 0.5 * problem.head[j].y.lo + 0.5 * problem.head[j].y.hi;
        let mut row = Vec::with_capacity(samples);
        for (k, &feature) in features.iter().enumerate() {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            let a = weights[j * width + feature];
            b -= a * centers[k];
            row.push(a);
        }
        if !good(b) {
            return None;
        }
        if row.iter().any(|v| !good(*v)) {
            return None;
        }
        projected.push(row);
        projected_bias.push(b);
    }
    for j in 0..heads {
        if !live(deadline) {
            return None;
        }
        let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
        if group.contains(&j) {
            continue;
        }
        if q > 0.0 {
            let variable = request.bounds.len();
            let projected_upper = projected[j]
                .iter()
                .zip(&boxes)
                .fold(projected_bias[j], |sum, (&a, x)| {
                    sum + a * if a >= 0.0 { x.hi } else { x.lo }
                });
            if !good(projected_upper) {
                return None;
            }
            request
                .bounds
                .push([0.0, projected_upper.max(problem.head[j].y.hi).max(0.0)]);
            request.objective.push(q);
            request.rows.push(HeadFunctionalLpRow {
                coefficients: vec![(variable, 1.0)],
                lower: 0.0,
            });
            let mut coefficients: Vec<_> = projected[j]
                .iter()
                .enumerate()
                .filter(|(_, a)| **a != 0.0)
                .map(|(k, &a)| (k, -a))
                .collect();
            coefficients.push((variable, 1.0));
            request.slope_rows[j] = Some(request.rows.len());
            request.rows.push(HeadFunctionalLpRow {
                coefficients,
                lower: projected_bias[j],
            });
        } else {
            for (k, &a) in projected[j].iter().enumerate() {
                request.objective[k] += baseline[j] * a;
            }
        }
    }
    for (i, &j) in group.iter().enumerate() {
        if !live(deadline) {
            return None;
        }
        let s = f64::from(u8::from(mask & (1 << i) != 0));
        let eta = 2.0 * s - 1.0;
        let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
        for (k, &a) in projected[j].iter().enumerate() {
            request.objective[k] += q * s * a;
        }
        let coefficients = projected[j]
            .iter()
            .enumerate()
            .filter(|(_, a)| **a != 0.0)
            .map(|(k, &a)| (k, eta * a))
            .collect();
        request.phase_rows.push(request.rows.len());
        request.rows.push(HeadFunctionalLpRow {
            coefficients,
            lower: -eta * projected_bias[j],
        });
    }
    (live(deadline) && request.validate()).then_some(request)
}

struct Attempt {
    hint: Option<(HeadFunctionalLpRequest, HeadFunctionalLpHint)>,
    baseline: Option<f64>,
    report: AdviceRefinement,
    supported: bool,
}

fn ask<F>(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    mask: usize,
    budget: &mut AdviceBudget,
    deadline: Instant,
    preparation_start: Option<Instant>,
    provider: F,
) -> Option<Attempt>
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let begin = preparation_start.unwrap_or_else(Instant::now);
    ask_with_elapsed(
        eng,
        pack,
        problem,
        group,
        mask,
        budget,
        deadline,
        begin,
        provider,
        || begin.elapsed(),
    )
}

// A supplied elapsed clock makes every early-return accounting branch
// deterministic in tests. Production always supplies Instant::elapsed.
fn ask_with_elapsed<F, E>(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    mask: usize,
    budget: &mut AdviceBudget,
    deadline: Instant,
    begin: Instant,
    provider: F,
    elapsed: E,
) -> Option<Attempt>
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
    E: FnOnce() -> Duration,
{
    let outcome = (|| {
        let mut attempt = Attempt {
            hint: None,
            baseline: None,
            report: AdviceRefinement::default(),
            supported: false,
        };
        let Some(call_deadline) = budget.call_deadline(begin, deadline) else {
            return Some(attempt);
        };
        if !live(call_deadline) {
            return Some(attempt);
        }
        if group.is_empty() {
            // Complete the one-mask owned cover BEFORE lazy construction.
            let Some(lower) = unconditional_baseline(problem, call_deadline) else {
                return Some(attempt);
            };
            attempt.baseline = Some(lower);
        }
        // A selected caller has already completed its complete owned cover.
        let backend = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&provider))
            .ok()
            .flatten();
        let Some(backend) = backend else {
            return Some(attempt);
        };
        let capable = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.supports_functional_advice()
        }))
        .unwrap_or(false);
        if !fp_qualified() || !pack.owned_for_head_groups(eng) {
            return None;
        }
        // Unsupported providers retain both V1 call slots. Their setup time
        // still reduces the shared 250ms wall allowance.
        if !capable {
            return Some(attempt);
        }
        attempt.supported = true;
        budget.calls += 1;
        let request = export_functional_lp(eng, problem, group, mask, call_deadline);
        if let Some(request) = request.filter(|_| live(call_deadline)) {
            attempt.report.calls += 1;
            let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.propose_functional(&request, call_deadline)
            }))
            .ok()
            .flatten();
            if let Some(hint) = hint.filter(|h| h.validate(&request) && live(call_deadline)) {
                attempt.hint = Some((request, hint));
            }
        }
        if !fp_qualified() || !pack.owned_for_head_groups(eng) {
            return None;
        }
        Some(attempt)
    })();
    // Includes baseline/preparation, factory and capability faults, export
    // failures, declined proposals and expiry, on EVERY return path.
    budget.charge_elapsed(elapsed());
    outcome
}

fn candidates(
    problem: &SourceHeadProblem,
    group: &[usize],
    mask: usize,
    request: &HeadFunctionalLpRequest,
    hint: &HeadFunctionalLpHint,
    deadline: Instant,
) -> Option<Vec<FunctionalProposal>> {
    if !hint.validate(request)
        || request.slope_rows.len() != problem.head.len()
        || request.phase_rows.len() != group.len()
    {
        return None;
    }
    let baseline = baseline_slopes(problem, group, deadline)?;
    let mut result = Vec::with_capacity(2);
    for sign in [1.0, -1.0] {
        let mut slopes = baseline.clone();
        for (j, role) in request.slope_rows.iter().enumerate() {
            if !live(deadline) {
                return None;
            }
            if let Some(r) = role {
                let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
                slopes[j] = (sign * hint.row_duals[*r]).max(0.0).min(q.max(0.0));
            }
        }
        let phase: Vec<_> = request
            .phase_rows
            .iter()
            .map(|&r| (sign * hint.row_duals[r]).max(0.0))
            .collect();
        if !result
            .iter()
            .any(|old: &FunctionalProposal| old.slopes == slopes && old.phase == phase)
        {
            result.push(FunctionalProposal {
                mask,
                slopes,
                phase,
            });
        }
    }
    Some(result)
}

fn transfer(
    problem: &SourceHeadProblem,
    group: &[usize],
    p: &FunctionalProposal,
    mask: usize,
) -> Option<FunctionalProposal> {
    let mut phase = Vec::with_capacity(group.len());
    for (i, &j) in group.iter().enumerate() {
        let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
        let old = f64::from(u8::from(p.mask & (1 << i) != 0));
        let s = f64::from(u8::from(mask & (1 << i) != 0));
        let mu = q * old - (2.0 * old - 1.0) * p.phase[i];
        let v = (2.0 * s - 1.0) * (q * s - mu);
        if !good(mu) || !good(v) {
            return None;
        }
        phase.push(v.max(0.0));
    }
    Some(FunctionalProposal {
        mask,
        slopes: p.slopes.clone(),
        phase,
    })
}

pub(super) struct PortfolioRefinement {
    pub lower: Vec<Option<f64>>,
    pub report: AdviceRefinement,
}

/// Visit the weakest still-open margins before any selected-group search.
/// At most two hints yield a bounded pool of normalized slope ADVICE. Each
/// recipient rebuilds its own point coefficients and EVERY full-Q/Y charge.
/// Six completed columns in two batches cover all three open CCT margins.
pub(super) fn refine_unconditional_portfolio(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &mut SourceHeadProblem,
    target: usize,
    adversaries: &[usize],
    existing: &[f64],
    budget: &mut AdviceBudget,
    deadline: Instant,
) -> Option<PortfolioRefinement> {
    if !advice::has_factory() {
        return Some(PortfolioRefinement {
            lower: vec![None; adversaries.len()],
            report: AdviceRefinement::default(),
        });
    }
    refine_portfolio_with_provider(
        eng,
        pack,
        problem,
        target,
        adversaries,
        existing,
        budget,
        deadline,
        advice::registered_provider,
    )
}

fn refine_portfolio_with_provider<F>(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &mut SourceHeadProblem,
    target: usize,
    adversaries: &[usize],
    existing: &[f64],
    budget: &mut AdviceBudget,
    deadline: Instant,
    provider: F,
) -> Option<PortfolioRefinement>
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    if existing.len() != adversaries.len() || !pack.owned_for_head_groups(eng) || !fp_qualified() {
        return None;
    }
    let mut output = PortfolioRefinement {
        lower: vec![None; adversaries.len()],
        report: AdviceRefinement::default(),
    };
    let mut rows: Vec<_> = (0..adversaries.len())
        .filter(|&r| existing[r] <= 0.0)
        .collect();
    rows.sort_by(|&a, &b| existing[a].total_cmp(&existing[b]).then(a.cmp(&b)));
    // Bound both native work and retained source-priced column storage. With
    // more than six open rows, the weakest six get one candidate each first.
    rows.truncate(6);
    let mut pool: Vec<Vec<Option<f64>>> = Vec::new();
    for &row in &rows {
        if !live(deadline) || budget.call_deadline(Instant::now(), deadline).is_none() {
            break;
        }
        let preparation_start = Instant::now();
        let Some(preparation_deadline) = budget.call_deadline(preparation_start, deadline) else {
            budget.charge_elapsed(preparation_start.elapsed());
            break;
        };
        if bind_margin(problem, eng, target, adversaries[row], preparation_deadline).is_none() {
            budget.charge_elapsed(preparation_start.elapsed());
            break;
        }
        if !problem.q.iter().any(|q| 0.5 * q.lo + 0.5 * q.hi > 0.0) {
            budget.charge_elapsed(preparation_start.elapsed());
            continue;
        }
        let attempt = ask(
            eng,
            pack,
            problem,
            &[],
            0,
            budget,
            deadline,
            Some(preparation_start),
            &provider,
        )?;
        output.report.calls += attempt.report.calls;
        if !attempt.supported {
            break;
        }
        if let Some((request, hint)) = attempt.hint {
            if let Some(proposals) = candidates(problem, &[], 0, &request, &hint, deadline) {
                for p in proposals {
                    let alpha: Vec<_> = request
                        .slope_rows
                        .iter()
                        .enumerate()
                        .map(|(j, role)| {
                            let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
                            role.and_then(|_| (q > 0.0).then(|| (p.slopes[j] / q).clamp(0.0, 1.0)))
                        })
                        .collect();
                    if !pool.contains(&alpha) {
                        pool.push(alpha);
                    }
                }
            }
        }
    }
    if pool.is_empty() || !live(deadline) {
        return Some(output);
    }
    let mut per_row: Vec<(usize, Vec<native::PreparedFunctional>)> = Vec::new();
    for &row in &rows {
        if !live(deadline) {
            break;
        }
        bind_margin(problem, eng, target, adversaries[row], deadline)?;
        let Some(base) = baseline_slopes(problem, &[], deadline) else {
            break;
        };
        let mut proposals: Vec<(f64, FunctionalProposal)> = Vec::new();
        for alpha in &pool {
            let mut slopes = base.clone();
            for (j, a) in alpha.iter().enumerate() {
                let q = 0.5 * problem.q[j].lo + 0.5 * problem.q[j].hi;
                if q > 0.0 {
                    if let Some(a) = a {
                        slopes[j] = q * a;
                    }
                }
            }
            if proposals.iter().any(|(_, p)| p.slopes == slopes) {
                continue;
            }
            if let Some(score) = native::score_functional(eng, problem, &slopes, deadline) {
                proposals.push((
                    score,
                    FunctionalProposal {
                        mask: 0,
                        slopes,
                        phase: vec![],
                    },
                ));
            }
        }
        proposals.sort_by(|(a, _), (b, _)| b.total_cmp(a));
        let selected: Vec<_> = proposals.into_iter().take(2).map(|(_, p)| p).collect();
        if selected.is_empty() {
            continue;
        }
        if let Some(prepared) =
            native::prepare_functionals(eng, pack, problem, &[], &selected, deadline)
        {
            per_row.push((row, prepared));
        }
    }
    let mut portfolio: Vec<(usize, &native::PreparedFunctional)> = Vec::new();
    for rank in 0..2 {
        for (row, proposals) in &per_row {
            if let Some(p) = proposals.get(rank) {
                portfolio.push((*row, p));
            }
        }
    }
    portfolio.truncate(6);
    for batch in portfolio
        .chunks(MAX_NATIVE_COLUMNS)
        .take(MAX_NATIVE_BATCHES)
    {
        if !live(deadline) {
            break;
        }
        let prepared: Vec<_> = batch.iter().map(|(_, p)| *p).collect();
        output.report.candidates += prepared.len();
        if let Some(replayed) =
            native::replay_prepared_functional_refs(eng, pack, problem, &prepared, deadline)
        {
            output.report.native_passes += replayed.passes;
            output.report.native_columns += replayed.columns;
            for (index, lower) in replayed.bounds {
                let row = batch[index].0;
                output.lower[row] = Some(output.lower[row].map_or(lower, |b| b.max(lower)));
            }
        }
        if !fp_qualified() || !pack.owned_for_head_groups(eng) {
            return None;
        }
    }
    Some(output)
}

#[cfg(test)]
fn refine_unconditional_with_provider<F>(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    budget: &mut AdviceBudget,
    deadline: Instant,
    provider: F,
) -> Option<UnconditionalRefinement>
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let mut output = UnconditionalRefinement::default();
    if !fp_qualified() || !pack.owned_for_head_groups(eng) {
        return None;
    }
    if !live(deadline) || budget.call_deadline(Instant::now(), deadline).is_none() {
        return Some(output);
    }
    let attempt = ask(eng, pack, problem, &[], 0, budget, deadline, None, provider)?;
    output.report = attempt.report;
    let Some(baseline) = attempt.baseline else {
        return Some(output);
    };
    let Some((request, hint)) = attempt.hint else {
        return Some(output);
    };
    let Some(proposals) = candidates(problem, &[], 0, &request, &hint, deadline) else {
        return Some(output);
    };
    output.report.candidates = proposals.len();
    if let Some(replayed) =
        native::replay_functionals(eng, pack, problem, &[], &proposals, deadline)
    {
        output.report.native_passes += replayed.passes;
        output.report.native_columns += replayed.columns;
        output.lower = Some(
            replayed
                .bounds
                .into_iter()
                .map(|(_, b)| b)
                .fold(baseline, f64::max),
        );
    }
    if !fp_qualified() || !pack.owned_for_head_groups(eng) {
        return None;
    }
    Some(output)
}

pub(super) fn refine_complete_cover(
    eng: &BackwardEngine<'_>,
    pack: &YPack,
    problem: &SourceHeadProblem,
    group: &[usize],
    cover: &mut RootGroupBound<'_>,
    budget: &mut AdviceBudget,
    deadline: Instant,
) -> Option<AdviceRefinement> {
    if !advice::has_factory() {
        return Some(AdviceRefinement::default());
    }
    refine_cover_with_provider(
        eng,
        pack,
        problem,
        group,
        cover,
        budget,
        deadline,
        advice::registered_provider,
    )
}

fn refine_cover_with_provider<F>(
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
    if !cover.applies_to(problem)
        || cover.group != group
        || cover.phases != 1 << group.len()
        || cover.phase_lower.len() != cover.phases
        || cover.proposals.len() != cover.phases
        || cover.phase_lower.iter().any(|x| !good(*x))
        || !fp_qualified()
        || !pack.owned_for_head_groups(eng)
    {
        return None;
    }
    let mut report = AdviceRefinement::default();
    if !live(deadline) || cover.lower > 0.0 {
        return Some(report);
    }
    let Some(mask) = (0..cover.phases)
        .filter(|&m| cover.phase_lower[m] <= 0.0)
        .min_by(|&a, &b| {
            cover.phase_lower[a]
                .total_cmp(&cover.phase_lower[b])
                .then(a.cmp(&b))
        })
    else {
        return Some(report);
    };
    let attempt = ask(
        eng, pack, problem, group, mask, budget, deadline, None, provider,
    )?;
    report.calls = attempt.report.calls;
    let Some((request, hint)) = attempt.hint else {
        return Some(report);
    };
    let Some(origins) = candidates(problem, group, mask, &request, &hint, deadline) else {
        return Some(report);
    };
    let before = cover.phase_lower.clone();
    let mut masks: Vec<_> = (0..cover.phases)
        .filter(|&m| cover.phase_lower[m] <= 0.0)
        .collect();
    masks.sort_by(|&a, &b| {
        cover.phase_lower[a]
            .total_cmp(&cover.phase_lower[b])
            .then(a.cmp(&b))
    });
    let mut portfolio = Vec::new();
    // Breadth first across still-open masks, then the other dual convention.
    for origin in &origins {
        for &other in &masks {
            if let Some(p) = transfer(problem, group, origin, other) {
                portfolio.push(p);
            }
        }
    }
    for proposals in portfolio
        .chunks(MAX_NATIVE_COLUMNS)
        .take(MAX_NATIVE_BATCHES)
    {
        if !live(deadline) {
            break;
        }
        report.candidates += proposals.len();
        if let Some(replayed) =
            native::replay_functionals(eng, pack, problem, group, proposals, deadline)
        {
            report.native_passes += replayed.passes;
            report.native_columns += replayed.columns;
            for (index, lower) in replayed.bounds {
                let p = &proposals[index];
                if lower > cover.phase_lower[p.mask] {
                    cover.phase_lower[p.mask] = lower;
                    cover.proposals[p.mask] = Proposal {
                        mask: p.mask,
                        phase: p.phase.clone(),
                    };
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
        if cover.lower > 0.0 {
            break;
        }
    }
    report.improved_phases = before
        .iter()
        .zip(&cover.phase_lower)
        .filter(|(a, b)| b > a)
        .count();
    Some(report)
}

#[cfg(test)]
#[path = "head_group_functional_tests.rs"]
mod tests;
