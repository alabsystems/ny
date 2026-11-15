// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::bias::{add_f64_down, f64_to_f32_down_no_subnormal, f64_to_f32_up_no_subnormal};
use super::*;
use crate::LinearBounds;

/// A positive target-minus-competitor margin at one explicit position.
/// IDs/scope/full coverage are additionally enforced by the prepared caller.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LogitDifference {
    pub target: usize,
    pub competitor: usize,
    pub position: usize,
}

pub(crate) fn sparse_request(
    outputs: usize,
    inputs: usize,
    rows: usize,
    positions: usize,
) -> Result<WorkRequest> {
    OriginalAffineView::validation_request(outputs, inputs, false)?;
    if positions == 0 {
        return Err(invalid("sparse positions must be nonempty"));
    }
    let columns = mul(positions, inputs)?;
    let outputs = cap_elements(mul(rows, columns)?)?;
    let elements = add(mul(outputs, 4)?, mul(rows, 2)?)?;
    output_request(
        elements,
        add(
            add(mul(mul(rows, inputs)?, 64)?, mul(outputs, 8)?)?,
            mul(rows, 64)?,
        )?,
    )
}

/// Seed directly over the terminal head INPUT. No rows-by-vocabulary matrix
/// is allocated. All other positions remain exact-zero and error-free.
pub(crate) fn seed_logit_differences(
    parameters: OriginalAffineView<'_>,
    positions: usize,
    differences: &[LogitDifference],
    phase: &mut OriginalAffinePhase,
) -> Result<LinearBounds> {
    phase.entry()?;
    sparse_request(
        parameters.out_features(),
        parameters.in_features(),
        differences.len(),
        positions,
    )?;
    for difference in differences {
        phase.spend(8)?;
        if difference.position >= positions
            || difference.target >= parameters.out_features()
            || difference.competitor >= parameters.out_features()
        {
            return Err(invalid("sparse original logit ID/position"));
        }
    }
    let rows = differences.len();
    let n = parameters.in_features();
    let columns = mul(positions, n)?;
    let len = mul(rows, columns)?;
    phase.retain_bytes(1024)?;
    let mut la = phase.output_vec(len)?;
    let mut ua = phase.output_vec(len)?;
    let mut le = phase.output_vec(len)?;
    let mut ue = phase.output_vec(len)?;
    let mut lb = phase.output_vec(rows)?;
    let mut ub = phase.output_vec(rows)?;
    for (row, difference) in differences.iter().enumerate() {
        for j in 0..n {
            phase.spend(64)?;
            let target = f32_to_f64_exact(parameters.weight[[difference.target, j]]);
            let competitor = f32_to_f64_exact(parameters.weight[[difference.competitor, j]]);
            let lower = add_f64_down(target, -competitor);
            let upper = add_f64_up(target, -competitor);
            let (coefficient, error) = stored_coefficient(lower, upper, 0.0)?;
            let index = row * columns + difference.position * n + j;
            la[index] = coefficient;
            ua[index] = coefficient;
            le[index] = error;
            ue[index] = error;
        }
        phase.spend(16)?;
        let target = parameters.bias_at(difference.target);
        let competitor = parameters.bias_at(difference.competitor);
        lb[row] = f64_to_f32_down_no_subnormal(add_f64_down(target, -competitor));
        ub[row] = f64_to_f32_up_no_subnormal(add_f64_up(target, -competitor));
    }
    let result = LinearBounds::from_prevalidated_parts_with_optional_err(
        array2(rows, columns, la)?,
        ndarray::Array1::from_vec(lb),
        array2(rows, columns, ua)?,
        ndarray::Array1::from_vec(ub),
        Some(array2(rows, columns, le)?),
        Some(array2(rows, columns, ue)?),
    )?;
    phase.checkpoint()?;
    Ok(result)
}
