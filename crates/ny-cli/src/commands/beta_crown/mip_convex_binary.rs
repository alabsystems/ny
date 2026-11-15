// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Optional integrality relaxation of convex heads in one signed margin.
//!
//! Every original column bound and row survives. Therefore every original
//! unsafe graph point remains feasible; only checked infeasibility of THIS
//! modified model may prove safety. A feasible point still needs the existing
//! original-network counterexample replay. This module grants no verdict.

use ny_core::Bound;
use ny_mip::ir::{Col, Row};
use ny_mip::MipParts;
use ny_onnx::vnnlib::VnnLibSpec;
use std::time::Instant;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct RelaxationReport {
    pub(super) unstable_before: usize,
    pub(super) relaxed_nonnegative: usize,
    pub(super) negative_binaries: usize,
}

struct Plan {
    relax: Vec<Col>,
    keep: Vec<Col>,
    widths: Vec<f64>,
}

/// Plan all changes before touching the problem or its branching metadata.
#[allow(clippy::too_many_arguments)]
pub(super) fn relax(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    bounds: &[Vec<Bound>],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    deadline: Instant,
) -> Option<RelaxationReport> {
    if !single_hidden || Instant::now() >= deadline || spec.dual_network.is_some() {
        return None;
    }
    let (first, second, reverse) = super::certified_shared_tree_pairwise_indices(spec)?;
    let [d, h, o] = dims else { return None };
    if *d == 0
        || *h == 0
        || *o != spec.num_outputs
        || *d != spec.num_inputs
        || weights.len() != 2
        || biases.len() != 2
        || bounds.len() != 1
        || weights[0].len() != d.checked_mul(*h)?
        || weights[1].len() != h.checked_mul(*o)?
        || biases[0].len() != *h
        || biases[1].len() != *o
        || bounds[0].len() != *h
        || weights
            .iter()
            .chain(biases.iter())
            .flatten()
            .any(|v| !v.is_finite())
        || parts.input_vars != (0..*d).map(Col).collect::<Vec<_>>()
        || parts.num_cols != parts.problem.num_cols()
        || parts.output_vars.len() != *o
    {
        return None;
    }
    check_unsafe_row(parts, unsafe_rows, first, second, reverse)?;
    let (positive, negative) = if reverse {
        (second, first)
    } else {
        (first, second)
    };
    let plan = mapped_plan(
        parts,
        *d,
        *h,
        *o,
        &bounds[0],
        &weights[1],
        positive,
        negative,
    )?;
    if Instant::now() >= deadline || !parts.problem.relax_binary_integrality(&plan.relax) {
        return None;
    }
    let report = RelaxationReport {
        unstable_before: parts.binary_vars.len(),
        relaxed_nonnegative: plan.relax.len(),
        negative_binaries: plan.keep.len(),
    };
    parts.binary_vars = plan.keep;
    parts.binary_widths = plan.widths;
    Some(report)
}

fn check_unsafe_row(
    parts: &MipParts,
    rows: &[Row],
    first: usize,
    second: usize,
    reverse: bool,
) -> Option<()> {
    let [row] = rows else { return None };
    let actual = parts.problem.rows().get(row.0)?;
    let expected = vec![
        (parts.output_vars.get(first)?.0, 1.0),
        (parts.output_vars.get(second)?.0, -1.0),
    ];
    let (lb, ub) = if reverse {
        (0.0, f64::INFINITY)
    } else {
        (f64::NEG_INFINITY, 0.0)
    };
    (actual.coeffs == expected && actual.lb == lb && actual.ub == ub).then_some(())
}

/// Reconstruct the encoder's complete one-layer ReLU column creation order.
/// Comparisons of exact finite output coefficients determine the sign without
/// rounding a subtraction; zero coefficients may also be relaxed safely.
#[allow(clippy::too_many_arguments)]
fn mapped_plan(
    parts: &MipParts,
    d: usize,
    h: usize,
    o: usize,
    bounds: &[Bound],
    output: &[f64],
    positive: usize,
    negative: usize,
) -> Option<Plan> {
    let mut next = d.checked_add(h)?;
    let mut binary = 0usize;
    let mut plan = Plan {
        relax: Vec::new(),
        keep: Vec::new(),
        widths: Vec::new(),
    };
    for (j, bound) in bounds.iter().enumerate() {
        let (l, u) = (f64::from(bound.lower()), f64::from(bound.upper()));
        if !l.is_finite() || !u.is_finite() || l > u {
            return None;
        }
        if l >= 0.0 {
            continue;
        }
        if u <= 0.0 {
            next = next.checked_add(1)?;
            continue;
        }
        let expected = Col(next.checked_add(1)?);
        let col = *parts.binary_vars.get(binary)?;
        let width = *parts.binary_widths.get(binary)?;
        let actual = parts.problem.cols().get(col.0)?;
        if col != expected
            || !actual.integer
            || actual.lb != 0.0
            || actual.ub != 1.0
            || width.to_bits() != (u - l).to_bits()
        {
            return None;
        }
        if output[positive * h + j] >= output[negative * h + j] {
            plan.relax.push(col);
        } else {
            plan.keep.push(col);
            plan.widths.push(width);
        }
        binary = binary.checked_add(1)?;
        next = next.checked_add(2)?;
    }
    if binary != parts.binary_vars.len()
        || binary != parts.binary_widths.len()
        || parts.problem.cols().iter().filter(|c| c.integer).count() != binary
        || parts.output_vars != (next..next.checked_add(o)?).map(Col).collect::<Vec<_>>()
        || next.checked_add(o)? != parts.num_cols
    {
        return None;
    }
    Some(plan)
}

#[cfg(test)]
#[path = "mip_convex_binary_tests.rs"]
mod tests;
