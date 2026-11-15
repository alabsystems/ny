// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Multiplier search only. Exactly one selected hint gets Source replay.
use super::*;

fn greedy_hint(
    model: &triangle::Model<'_>,
    cuts: &[CertifiedDirection<'_>],
    order: &[usize],
) -> Option<HeadFunctionalLpHint> {
    let request = model.request();
    let first = request.rows.len().checked_sub(cuts.len())?;
    let mut residual = request.objective.clone();
    let mut row_duals = vec![0.0; request.rows.len()];
    for &at in order {
        // Every signed head term spends original objective mass. Positive
        // coefficients cap the same multiplier instead of being ignored.
        let lambda = request.rows[first + at]
            .coefficients
            .iter()
            .filter(|&&(j, a)| j >= request.feature_count && a != 0.0)
            .map(|&(j, a)| {
                if a < 0.0 {
                    (-residual[j]).max(0.0) / -a
                } else {
                    residual[j].max(0.0) / a
                }
            })
            .fold(f64::INFINITY, f64::min);
        if !good(lambda) {
            return None;
        }
        row_duals[first + at] = lambda;
        for &(j, a) in &request.rows[first + at].coefficients {
            residual[j] -= lambda * a;
        }
    }
    // The original objective includes stable heads and every original input.
    for (head, &role) in request.slope_rows.iter().enumerate() {
        let upper = role?.checked_add(1)?;
        if upper >= model.scalar_count() {
            continue;
        }
        let row = &request.rows[upper];
        let Some(&(_, a)) = row
            .coefficients
            .iter()
            .find(|&&(j, a)| j == request.feature_count + head && a < 0.0)
        else {
            continue;
        };
        let lambda = (-residual[request.feature_count + head]).max(0.0) / -a;
        if !good(lambda) {
            return None;
        }
        row_duals[upper] = lambda;
        for &(j, a) in &row.coefficients {
            residual[j] -= lambda * a;
        }
    }
    let hint = HeadFunctionalLpHint { row_duals };
    hint.validate(request).then_some(hint)
}

/// Exported midpoint arithmetic ranks multiplier proposals; its endpoint is
/// never a bound. The selected vector still receives ordinary interval replay.
fn score(request: &HeadFunctionalLpRequest, hint: &HeadFunctionalLpHint) -> Option<f64> {
    let mut residual = request.objective.clone();
    let mut value = 0.0;
    for (row, &lambda) in request.rows.iter().zip(&hint.row_duals) {
        value += lambda * row.lower;
        for &(j, a) in &row.coefficients {
            residual[j] -= lambda * a;
        }
    }
    for (&a, &[lo, hi]) in residual.iter().zip(&request.bounds) {
        value += a * if a < 0.0 { hi } else { lo };
    }
    good(value).then_some(value)
}

fn alignment(
    cut: &CertifiedDirection<'_>,
    gradient: &[f64],
    request: &HeadFunctionalLpRequest,
) -> f64 {
    let mut score = 0.0;
    for &(j, a) in &cut.eta {
        score += a * gradient[j] * (request.bounds[j][1] - request.bounds[j][0]);
    }
    if good(score) {
        score
    } else {
        0.0
    }
}

pub(super) fn portfolio_hint(
    model: &triangle::Model<'_>,
    cuts: &[CertifiedDirection<'_>],
    gradient: &[f64],
    budget: &mut Budget,
) -> Option<HeadFunctionalLpHint> {
    let request = model.request();
    if gradient.len() != request.feature_count {
        return None;
    }
    let mut order: Vec<_> = (0..cuts.len()).collect();
    let mut best = greedy_hint(model, cuts, &order)?;
    let mut best_score = score(request, &best);
    // Admission already charged the insertion-order proposal. This optional
    // alternate order has its own weighted search/replay-admission credits.
    let terms = request
        .rows
        .iter()
        .try_fold(request.bounds.len(), |n, row| {
            n.checked_add(row.coefficients.len())
        })?;
    order.sort_by(|&a, &b| {
        alignment(&cuts[b], gradient, request)
            .total_cmp(&alignment(&cuts[a], gradient, request))
            .then(a.cmp(&b))
    });
    // Scalar-only is a genuine fallback: nearly redundant geometry can make
    // greedy head cancellation worse than the original scalar chords.
    for candidate_order in [vec![], order] {
        if budget.charge(terms.checked_mul(2)?).is_none() {
            break;
        }
        if let Some(candidate) = greedy_hint(model, cuts, &candidate_order) {
            if let Some(value) = score(request, &candidate) {
                if best_score.is_none_or(|old| value > old) {
                    best = candidate;
                    best_score = Some(value);
                }
            }
        }
    }
    Some(best)
}
