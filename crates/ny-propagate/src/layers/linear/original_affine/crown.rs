// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::bias::{add_f64_down, f64_to_f32_down_no_subnormal, f64_to_f32_up_no_subnormal};
use super::*;
use crate::LinearBounds;

fn geometry(
    outputs: usize,
    inputs: usize,
    rows: usize,
    columns: usize,
) -> Result<(usize, usize, usize)> {
    OriginalAffineView::validation_request(outputs, inputs, false)?;
    if columns == 0 || !columns.is_multiple_of(outputs) {
        return Err(invalid("CROWN position/output feature mismatch"));
    }
    let positions = columns / outputs;
    let incoming = cap_elements(mul(rows, columns)?)?;
    let target_columns = mul(positions, inputs)?;
    let outgoing = cap_elements(mul(rows, target_columns)?)?;
    Ok((incoming, outgoing, positions))
}

pub(crate) fn crown_request(
    outputs: usize,
    inputs: usize,
    rows: usize,
    columns: usize,
) -> Result<WorkRequest> {
    let (incoming, outgoing, _) = geometry(outputs, inputs, rows, columns)?;
    let macs = mul(outgoing, outputs)?;
    let elements = add(mul(outgoing, 4)?, mul(rows, 2)?)?;
    output_request(
        elements,
        add(
            add(mul(macs, 48)?, mul(incoming, 32)?)?,
            add(mul(outgoing, 48)?, mul(rows, 64)?)?,
        )?,
    )
}

/// Compose signed lower/upper carriers with their complete uncertainty planes.
/// E*|W| remains coefficient uncertainty; E*|bias| is subtracted/added to the
/// corresponding bias. Both storage casts and all binary64 sums are paid.
pub(crate) fn crown(
    parameters: OriginalAffineView<'_>,
    incoming: &LinearBounds,
    phase: &mut OriginalAffinePhase,
) -> Result<LinearBounds> {
    phase.entry()?;
    let (rows, columns) = incoming.lower_a().dim();
    let (_, output_len, positions) = geometry(
        parameters.out_features(),
        parameters.in_features(),
        rows,
        columns,
    )?;
    if incoming.upper_a().dim() != (rows, columns)
        || incoming.lower_b().len() != rows
        || incoming.upper_b().len() != rows
        || incoming
            .lower_a_err()
            .is_some_and(|e| e.dim() != (rows, columns))
        || incoming
            .upper_a_err()
            .is_some_and(|e| e.dim() != (rows, columns))
    {
        return Err(invalid("CROWN incoming carrier shapes"));
    }
    for &value in incoming
        .lower_a()
        .iter()
        .chain(incoming.upper_a().iter())
        .chain(incoming.lower_b().iter())
        .chain(incoming.upper_b().iter())
    {
        phase.spend(4)?;
        if !value.is_finite() {
            return Err(numeric("nonfinite incoming carrier/bias"));
        }
    }
    for &value in incoming
        .lower_a_err()
        .iter()
        .chain(incoming.upper_a_err().iter())
        .flat_map(|e| e.iter())
    {
        phase.spend(4)?;
        incoming_error(value)?;
    }
    phase.retain_bytes(1024)?;
    let mut la = phase.output_vec(output_len)?;
    let mut ua = phase.output_vec(output_len)?;
    let mut le = phase.output_vec(output_len)?;
    let mut ue = phase.output_vec(output_len)?;
    let mut lb = phase.output_vec(rows)?;
    let mut ub = phase.output_vec(rows)?;
    let n = parameters.in_features();
    let m = parameters.out_features();
    let target_columns = mul(positions, n)?;
    for row in 0..rows {
        let mut lower_bias = f32_to_f64_exact(incoming.lower_b()[row]);
        let mut upper_bias = f32_to_f64_exact(incoming.upper_b()[row]);
        for position in 0..positions {
            for k in 0..m {
                phase.spend(16)?;
                let column = position * m + k;
                let a = f32_to_f64_exact(incoming.lower_a()[[row, column]]);
                let b = f32_to_f64_exact(incoming.upper_a()[[row, column]]);
                let e = incoming
                    .lower_a_err()
                    .map_or(0.0, |e| f32_to_f64_exact(e[[row, column]]));
                let f = incoming
                    .upper_a_err()
                    .map_or(0.0, |e| f32_to_f64_exact(e[[row, column]]));
                let bias = parameters.bias_at(k);
                lower_bias = add_f64_down(lower_bias, a * bias);
                lower_bias = add_f64_down(lower_bias, -(e * bias.abs()));
                upper_bias = add_f64_up(upper_bias, b * bias);
                upper_bias = add_f64_up(upper_bias, f * bias.abs());
            }
            for start in (0..n).step_by(TILE) {
                phase.checkpoint()?;
                let width = (n - start).min(TILE);
                let (mut llo, mut lhi, mut ulo, mut uhi, mut el, mut eu) = (
                    [0.0; TILE],
                    [0.0; TILE],
                    [0.0; TILE],
                    [0.0; TILE],
                    [0.0; TILE],
                    [0.0; TILE],
                );
                for k in 0..m {
                    let column = position * m + k;
                    let a = f32_to_f64_exact(incoming.lower_a()[[row, column]]);
                    let b = f32_to_f64_exact(incoming.upper_a()[[row, column]]);
                    let e = incoming
                        .lower_a_err()
                        .map_or(0.0, |e| f32_to_f64_exact(e[[row, column]]));
                    let f = incoming
                        .upper_a_err()
                        .map_or(0.0, |e| f32_to_f64_exact(e[[row, column]]));
                    for j in 0..width {
                        phase.spend(48)?;
                        let w = f32_to_f64_exact(parameters.weight[[k, start + j]]);
                        let x = a * w;
                        let y = b * w;
                        llo[j] = add_f64_down(llo[j], x);
                        lhi[j] = add_f64_up(lhi[j], x);
                        ulo[j] = add_f64_down(ulo[j], y);
                        uhi[j] = add_f64_up(uhi[j], y);
                        el[j] = add_f64_up(el[j], e * w.abs());
                        eu[j] = add_f64_up(eu[j], f * w.abs());
                    }
                }
                for j in 0..width {
                    phase.spend(32)?;
                    let index = row * target_columns + position * n + start + j;
                    (la[index], le[index]) = stored_coefficient(llo[j], lhi[j], el[j])?;
                    (ua[index], ue[index]) = stored_coefficient(ulo[j], uhi[j], eu[j])?;
                }
            }
        }
        phase.spend(16)?;
        lb[row] = f64_to_f32_down_no_subnormal(lower_bias);
        ub[row] = f64_to_f32_up_no_subnormal(upper_bias);
    }
    let result = LinearBounds::from_prevalidated_parts_with_optional_err(
        array2(rows, target_columns, la)?,
        ndarray::Array1::from_vec(lb),
        array2(rows, target_columns, ua)?,
        ndarray::Array1::from_vec(ub),
        Some(array2(rows, target_columns, le)?),
        Some(array2(rows, target_columns, ue)?),
    )?;
    phase.checkpoint()?;
    Ok(result)
}
