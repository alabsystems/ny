// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reuse an actual completed original-root vector. Only a published native-row
//! grant may spend one prepaid augmentation at the root before branching.
use super::*;

fn live(end: Instant) -> bool {
    Instant::now() < end && ny_core::instance_budget::deadline().is_none_or(|d| Instant::now() < d)
}

fn root_from_completed(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
    checked_lower: f64,
    end: Instant,
) -> Option<(Vec<(f64, f64)>, Vec<Role>, NodeOutcome)> {
    if !live(end) || !checked_lower.is_finite() {
        return None;
    }
    require_hull_environment().ok()?;
    let checked =
        super::super::certify_hull_lower_bound_polled(problem, multipliers, &mut || live(end))?;
    if checked.to_bits() != checked_lower.to_bits() {
        return None;
    }
    let bounds = preactivation_bounds(problem);
    let roles = classify(problem, &bounds);
    let triangle = triangle_multipliers(problem, &roles, &bounds);
    let triangle_lower = certify_hull_node_lower_bound(problem, &[], &triangle, &[]);
    let (value, m, route) = match triangle_lower {
        Some(value) if value > checked => (value, triangle, HullBoundRoute::Triangle),
        _ => (checked, multipliers.clone(), HullBoundRoute::Lp),
    };
    if !live(end) {
        return None;
    }
    require_hull_environment().ok()?;
    let phases = vec![None; problem.hidden_dim];
    let x_hat = node_minimizer(problem, &phases, &[], &m, &[]);
    let branch = choose_branch(problem, &roles, &bounds, &x_hat, &m);
    let warm = (m.alpha.len().checked_add(m.g.len())? <= WARM_MAX_CANDIDATE_CELLS).then_some(
        WarmCandidate {
            multipliers: m,
            beta: Vec::new(),
        },
    );
    if !live(end) {
        return None;
    }
    require_hull_environment().ok()?;
    Some((
        bounds,
        roles,
        NodeOutcome {
            certified: Some(value),
            cached: None,
            route: Some(route),
            lp_status: HullLpStatus::Skipped,
            lp_value: None,
            branch,
            warm,
        },
    ))
}

pub(in crate::single_hidden_hull) fn from_completed(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
    checked_lower: f64,
    end: Instant,
    options: HullBabOptions,
) -> Option<HullBabReport> {
    from_completed_with_cache(problem, multipliers, checked_lower, end, options, None)
}

pub(in crate::single_hidden_hull) fn from_completed_with_cache(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
    checked_lower: f64,
    end: Instant,
    options: HullBabOptions,
    cache: Option<&super::super::mip_cuts::SignedChildBound<'_>>,
) -> Option<HullBabReport> {
    from_completed_with_native(
        problem,
        multipliers,
        checked_lower,
        end,
        options,
        cache,
        None,
    )
}
pub(in crate::single_hidden_hull) fn from_completed_with_native(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
    checked_lower: f64,
    end: Instant,
    options: HullBabOptions,
    cache: Option<&super::super::mip_cuts::SignedChildBound<'_>>,
    native: Option<&super::super::mip_cuts::SignedNativeRows<'_>>,
) -> Option<HullBabReport> {
    let end = ny_core::instance_budget::deadline().map_or(end, |d| d.min(end));
    if cache.is_some_and(|cache| !cache.matches_problem(problem)) {
        return None;
    }
    let (bounds, roles, mut root) = root_from_completed(problem, multipliers, checked_lower, end)?;
    root.cached = cache.and_then(|cache| cache.root_lower());
    let calls = AtomicUsize::new(0);
    let root_before_native = native.map(|_| {
        root.certified
            .unwrap_or(f64::NEG_INFINITY)
            .max(root.cached.unwrap_or(f64::NEG_INFINITY))
    });
    if let Some(native) = native {
        improve_native_root(problem, &roles, &bounds, end, native, &calls, &mut root);
    }
    let root_lp_calls = calls.load(AtomicOrdering::Relaxed);
    let mut report = run_tree_with_cache_native(
        problem,
        end,
        options,
        &bounds,
        &roles,
        root,
        &calls,
        root_lp_calls,
        cache,
        native,
    );
    report.root_before_native = root_before_native;
    Some(report)
}

/// One optional root proposal, sharing the original two-augmentation bank with
/// children. This can add ONE total LP versus the reuse-only path: ordinary
/// child fallback calls are not bank credits. The base Model/session uses the
/// existing solver envelope; the bank pays incremental row/pricing storage.
/// Refusal never retries the already completed unaugmented root LP.
fn improve_native_root(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    end: Instant,
    native: &super::super::mip_cuts::SignedNativeRows<'_>,
    calls: &AtomicUsize,
    root: &mut NodeOutcome,
) {
    let end = end.min(native.end());
    let floor = root
        .certified
        .unwrap_or(f64::NEG_INFINITY)
        .max(root.cached.unwrap_or(f64::NEG_INFINITY));
    if floor > 0.0 || !live(end) || require_hull_environment().is_err() {
        return;
    }
    let Some((mut lp, split_rows)) = build_node_lp(problem, roles, bounds, &[]) else {
        return;
    };
    let old = |lp: &HullLp, y: &[f64]| -> Option<f64> {
        require_hull_environment().ok()?;
        let (m, beta) = node_multipliers_from_duals(problem, roles, lp, &split_rows, y)?;
        certify_hull_node_lower_bound(problem, &[], &m, &beta)
    };
    let run = |lp: &HullLp, certify: &mut dyn FnMut(&[f64]) -> Option<f64>| {
        let remaining = end.saturating_duration_since(Instant::now());
        let opts = SolveOpts::new()
            .with_time_limit(remaining.max(Duration::from_millis(1)))
            .with_deadline(end);
        guarded_hull_proposal(|| {
            calls.fetch_add(1, AtomicOrdering::Relaxed);
            propose_then_certify_min(&lp.model, &lp.objective, &opts, ReplayPolicy::Skip, certify)
        })
    };
    use super::super::mip_cuts::NativeVisit;
    let proposal = match native.with_node(
        problem,
        &[],
        &mut lp,
        &split_rows,
        end,
        &mut || live(end),
        old,
        run,
    ) {
        NativeVisit::Declined | NativeVisit::Stopped | NativeVisit::Committed(None) => return,
        NativeVisit::Committed(Some(proposal)) => proposal,
    };
    let outcome = match proposal {
        Err(_) => {
            root.lp_status = HullLpStatus::UnsupportedEnvironment;
            return;
        }
        Ok(Err(_)) => {
            root.lp_status = HullLpStatus::Error;
            return;
        }
        Ok(Ok(outcome)) => outcome,
    };
    root.lp_value = outcome.float_dual_value.map(|value| value + lp.constant);
    root.lp_status = if outcome.duals_available {
        HullLpStatus::NoCertificate
    } else {
        HullLpStatus::NoDuals
    };
    // The pricer rederives a bound from the full augmented dual on the original
    // root feasible set. Status and floating LP objective provide no authority.
    let Some(value) = outcome.closed_form.filter(|value| value.is_finite()) else {
        return;
    };
    if !outcome.duals_available || !live(end) {
        return;
    }
    if require_hull_environment().is_err() {
        root.lp_status = HullLpStatus::UnsupportedEnvironment;
        return;
    }
    root.lp_status = HullLpStatus::Solved;
    if value > floor {
        root.certified = Some(value);
        root.route = Some(HullBoundRoute::Lp);
    }
    // Keep the independently checked old warm vector and branch advice. A
    // scalar from the augmented pricer is not a legacy multiplier certificate.
}

#[cfg(test)]
#[path = "single_hidden_hull_bab_seed_tests.rs"]
mod tests;
