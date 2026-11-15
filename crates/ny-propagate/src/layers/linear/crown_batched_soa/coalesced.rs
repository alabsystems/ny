// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Coalesce independent domain rows through the shared certified CPU kernel.
//! There is no coefficient, bias, or error-bound arithmetic here. In
//! particular, do not substitute the old f32 multi-domain implementation.

use super::{LinearLayer, SoaLinearBackwardInput};
use crate::faer_parallelism::NestedFaerParGuard;
use crate::network::crown_memory::cpu_crown_dense_budget_bytes;
use crate::LinearBounds;
use ndarray::{s, Array1, Array2, ArrayView2};
use ny_core::{GemmEngine, NyError, Result};
use std::mem::size_of;
use std::time::Instant;

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CoalescedWork {
    pub shared_calls: usize,
    pub packed_domains: usize,
}

type DomainResults = Vec<Option<Result<LinearBounds>>>;

fn poll(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "coalesced Linear domains: original deadline exceeded".into(),
        ));
    }
    Ok(())
}

fn invalid(detail: &str) -> NyError {
    NyError::InvalidSpec(format!("coalesced Linear domains: {detail}"))
}

/// Conservative admission for staging the packed inputs and shared-kernel
/// outputs before scattering. It includes coefficient/error planes, biases,
/// widened product/copy intermediates and row padding allowance. The shared
/// kernel retains its own scratch/weight-cache guards as well. No memory
/// limit or numerical policy is widened when packing is declined.
fn staging_bytes(domains: usize, rows: usize, input: usize, output: usize) -> Option<usize> {
    domains.checked_mul(rows)?.checked_add(8)?.checked_mul(
        input
            .checked_mul(64)?
            .checked_add(output.checked_mul(128)?)?
            .checked_add(64)?,
    )
}

fn copied(
    values: impl IntoIterator<Item = f32>,
    length: usize,
    deadline: Option<Instant>,
) -> Result<Vec<f32>> {
    poll(deadline)?;
    let mut out = Vec::new();
    out.try_reserve_exact(length)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: length.saturating_mul(size_of::<f32>()),
            budget_bytes: cpu_crown_dense_budget_bytes(),
            site: "coalesced Linear domain copy",
        })?;
    for (i, value) in values.into_iter().enumerate() {
        if i % 4096 == 0 {
            poll(deadline)?;
        }
        if i >= length {
            return Err(invalid("copy exceeds its admitted shape"));
        }
        out.push(value);
    }
    poll(deadline)?;
    if out.len() != length {
        return Err(invalid("copy length disagrees with its shape"));
    }
    Ok(out)
}

fn matrix(
    values: impl IntoIterator<Item = f32>,
    rows: usize,
    cols: usize,
    deadline: Option<Instant>,
) -> Result<Array2<f32>> {
    let cells = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("matrix size overflow"))?;
    Array2::from_shape_vec((rows, cols), copied(values, cells, deadline)?)
        .map_err(|_| invalid("matrix shape disagrees with its copied payload"))
}

fn extract_matrix(view: ArrayView2<'_, f32>, deadline: Option<Instant>) -> Result<Array2<f32>> {
    matrix(view.iter().copied(), view.nrows(), view.ncols(), deadline)
}

fn extract_domain(
    bounds: &LinearBounds,
    start: usize,
    rows: usize,
    deadline: Option<Instant>,
) -> Result<LinearBounds> {
    let end = start
        .checked_add(rows)
        .ok_or_else(|| invalid("result row overflow"))?;
    let error = |a: Option<&Array2<f32>>| -> Result<Option<Array2<f32>>> {
        a.map(|a| extract_matrix(a.slice(s![start..end, ..]), deadline))
            .transpose()
    };
    // Intermediate nonfinite coefficients are valid inputs to the shared
    // conservative firewall. Preserve their representation without imposing
    // LinearBounds::new's stronger fresh-construction invariant.
    Ok(LinearBounds {
        lower_a: extract_matrix(bounds.lower_a().slice(s![start..end, ..]), deadline)?,
        upper_a: extract_matrix(bounds.upper_a().slice(s![start..end, ..]), deadline)?,
        lower_b: Array1::from_vec(copied(
            bounds.lower_b().slice(s![start..end]).iter().copied(),
            rows,
            deadline,
        )?),
        upper_b: Array1::from_vec(copied(
            bounds.upper_b().slice(s![start..end]).iter().copied(),
            rows,
            deadline,
        )?),
        lower_a_err: error(bounds.lower_a_err())?,
        upper_a_err: error(bounds.upper_a_err())?,
    })
}

/// Pack only flat Linear layouts. The caller keeps its historical per-domain
/// path for sequence layouts or a refused memory admission. Groups preserve
/// BOTH exact-marker Options: absent error planes are never replaced by zero
/// planes (0*nonfinite operands need not have the same conservative behavior).
pub(super) fn try_coalesced(
    layer: &LinearLayer,
    input: &SoaLinearBackwardInput<'_>,
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
    budget: usize,
) -> Result<Option<(DomainResults, CoalescedWork)>> {
    poll(deadline)?;
    if input.n_domains < 2 || input.rows == 0 || input.cols != layer.weight.nrows() {
        return Ok(None);
    }
    let domains = input.n_domains;
    let rc = input
        .rows
        .checked_mul(input.cols)
        .ok_or_else(|| invalid("coefficient shape overflow"))?;
    let coefficients = domains
        .checked_mul(rc)
        .ok_or_else(|| invalid("batch shape overflow"))?;
    let biases = domains
        .checked_mul(input.rows)
        .ok_or_else(|| invalid("bias shape overflow"))?;
    if input.active.len() != domains
        || input.lower_err_present.len() != domains
        || input.upper_err_present.len() != domains
        || input.lower_a.len() != coefficients
        || input.upper_a.len() != coefficients
        || input.lower_b.len() != biases
        || input.upper_b.len() != biases
        || input.lower_a_err.is_some_and(|p| p.len() != coefficients)
        || input.upper_a_err.is_some_and(|p| p.len() != coefficients)
    {
        return Err(invalid("plane or mask shape mismatch"));
    }
    let mut groups: [Vec<usize>; 4] = Default::default();
    for d in 0..domains {
        if d % 4096 == 0 {
            poll(deadline)?;
        }
        if !input.active[d] {
            continue;
        }
        if (input.lower_err_present[d] && input.lower_a_err.is_none())
            || (input.upper_err_present[d] && input.upper_a_err.is_none())
        {
            return Err(invalid(
                "an active domain's exact/error marker lacks its plane",
            ));
        }
        let key =
            usize::from(input.lower_err_present[d]) + 2 * usize::from(input.upper_err_present[d]);
        groups[key].push(d);
    }
    if !groups.iter().any(|group| group.len() > 1) {
        return Ok(None);
    }
    let active = groups.iter().map(Vec::len).sum();
    if staging_bytes(active, input.rows, input.cols, layer.weight.ncols())
        .is_none_or(|bytes| bytes > budget)
    {
        return Ok(None);
    }
    poll(deadline)?;
    let mut results: DomainResults = (0..domains).map(|_| None).collect();
    let mut work = CoalescedWork::default();
    for group in groups.iter().filter(|group| !group.is_empty()) {
        poll(deadline)?;
        let packed_rows = group
            .len()
            .checked_mul(input.rows)
            .ok_or_else(|| invalid("packed row overflow"))?;
        let packed_matrix = |plane: &[f32]| {
            matrix(
                group
                    .iter()
                    .flat_map(|&d| plane[d * rc..(d + 1) * rc].iter().copied()),
                packed_rows,
                input.cols,
                deadline,
            )
        };
        let packed_bias = |plane: &[f32]| -> Result<Array1<f32>> {
            let values = group
                .iter()
                .flat_map(|&d| plane[d * input.rows..(d + 1) * input.rows].iter().copied());
            Ok(Array1::from_vec(copied(values, packed_rows, deadline)?))
        };
        let error = |plane: Option<&[f32]>, present: bool| -> Result<Option<Array2<f32>>> {
            if present {
                packed_matrix(plane.ok_or_else(|| invalid("missing error plane"))?).map(Some)
            } else {
                Ok(None)
            }
        };
        let first = group[0];
        let packed = LinearBounds {
            lower_a: packed_matrix(input.lower_a)?,
            upper_a: packed_matrix(input.upper_a)?,
            lower_b: packed_bias(input.lower_b)?,
            upper_b: packed_bias(input.upper_b)?,
            lower_a_err: error(input.lower_a_err, input.lower_err_present[first])?,
            upper_a_err: error(input.upper_a_err, input.upper_err_present[first])?,
        };
        // Match the reference worker's faer policy. Every contraction, bias
        // operation and certified-error charge remains in the same function.
        let _faer_par = NestedFaerParGuard::new();
        let propagated = layer.propagate_linear_pollable_cpu(&packed, Some(engine), deadline)?;
        poll(deadline)?;
        let expected = (packed_rows, layer.weight.ncols());
        if propagated.lower_a().dim() != expected
            || propagated.upper_a().dim() != expected
            || propagated.lower_b().len() != packed_rows
            || propagated.upper_b().len() != packed_rows
            || propagated
                .lower_a_err()
                .is_some_and(|a| a.dim() != expected)
            || propagated
                .upper_a_err()
                .is_some_and(|a| a.dim() != expected)
        {
            return Err(invalid("shared kernel returned a malformed packed shape"));
        }
        work.shared_calls += 1;
        if group.len() > 1 {
            work.packed_domains += group.len();
        }
        for (offset, &d) in group.iter().enumerate() {
            let bounds = extract_domain(&propagated, offset * input.rows, input.rows, deadline)?;
            results[d] = Some(Ok(bounds));
        }
    }
    poll(deadline)?;
    Ok(Some((results, work)))
}

#[cfg(test)]
#[path = "coalesced/tests.rs"]
mod tests;
