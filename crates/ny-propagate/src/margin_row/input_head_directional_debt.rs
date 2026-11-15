// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Proposal-only signed geometry debt, never a certified residual.
use super::*;

/// Return signed objective debt, deliberately excluding scalar triangles from
/// the geometry residual. Full reduced cost may vanish with a negative floor.
pub(in super::super) fn geometry_debt(
    request: &HeadFunctionalLpRequest,
    scalar_rows: usize,
    geometry_start: usize,
    hint: &HeadFunctionalLpHint,
    sign: f64,
    budget: &mut Budget,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if !hint.validate(request)
        || ![1.0, -1.0].contains(&sign)
        || scalar_rows > geometry_start
        || geometry_start > request.rows.len()
    {
        return None;
    }
    let d = request.feature_count;
    let mut residual = request.objective.clone();
    for (row, &raw) in request
        .rows
        .iter()
        .zip(&hint.row_duals)
        .skip(geometry_start)
    {
        let lambda = (sign * raw).max(0.0);
        for &(j, a) in &row.coefficients {
            budget.charge(1)?;
            residual[j] -= lambda * a;
            if !good(residual[j]) {
                return None;
            }
        }
    }
    let mut debt = residual[d..].to_vec();
    let mut gradient = residual[..d].to_vec();
    for (head, (&role, q)) in request.slope_rows.iter().zip(&mut debt).enumerate() {
        budget.charge(1)?;
        let upper = role?.checked_add(1)?;
        if upper >= scalar_rows {
            continue;
        }
        let row = &request.rows[upper];
        let Some(&(_, coefficient)) = row
            .coefficients
            .iter()
            .find(|&&(j, a)| j == d + head && a < 0.0)
        else {
            continue;
        };
        // Alternate triangle-debt ranking protects against degenerate advice.
        // These rounded values are search data only, never a certificate.
        let triangle_debt = (sign * hint.row_duals[upper]).max(0.0) * -coefficient;
        if !good(triangle_debt) {
            return None;
        }
        *q = q.min(-triangle_debt);
        for &(j, a) in &row.coefficients {
            budget.charge(1)?;
            if j < d && *q < 0.0 {
                gradient[j] += *q * (-a / coefficient);
                if !good(gradient[j]) {
                    return None;
                }
            }
        }
    }
    Some((debt, gradient))
}
