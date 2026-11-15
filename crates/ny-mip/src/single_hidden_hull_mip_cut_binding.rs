// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact Source/encoder identity checks. None of the supplied layout is trusted.

use super::*;
use crate::ir::Col as IrCol;

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}
fn same_slice(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(*a, *b))
}

fn source_matches(
    cuts: &HullRootMipCuts<'_>,
    e: &HullMipEncoding<'_>,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<Bound>> {
    let p = cuts.problem;
    let [d, h, o] = e.dimensions else { return None };
    if *d != p.input_dim
        || *h != p.hidden_dim
        || *o == 0
        || e.first_output == e.second_output
        || e.first_output >= *o
        || e.second_output >= *o
        || e.weights.len() != 2
        || e.biases.len() != 2
        || e.hidden_bounds.len() != 1
        || e.weights[0].len() != d.checked_mul(*h)?
        || e.weights[1].len() != h.checked_mul(*o)?
        || e.weights[1].len() > ROOT_MIP_CUT_MAX_CELLS
        || e.biases[0].len() != *h
        || e.biases[1].len() != *o
        || e.hidden_bounds[0].len() != *h
        || !same_slice(&e.weights[0], &p.w1)
        || !same_slice(&e.biases[0], &p.b1)
        || e.weights
            .iter()
            .chain(e.biases.iter())
            .flatten()
            .any(|v| !v.is_finite())
        || e.hidden_bounds[0]
            .iter()
            .any(|b| !b.lower().is_finite() || !b.upper().is_finite() || b.lower() > b.upper())
        || !poll()
    {
        return None;
    }
    let objective = if e.reverse {
        [(e.second_output, 1.0), (e.first_output, -1.0)]
    } else {
        [(e.first_output, 1.0), (e.second_output, -1.0)]
    };
    let (c, d) = form_margin_down(*h, &e.weights[1], &e.biases[1], &objective, 0.0, poll).ok()?;
    if !same_slice(&c, &p.c) || !same(d, p.d) {
        return None;
    }
    let mut input = Vec::with_capacity(p.input_dim);
    for (&lo, &hi) in p.lower.iter().zip(&p.upper) {
        if !poll() {
            return None;
        }
        let (l, u) = (lo as f32, hi as f32);
        // This existing encoder uses binary32 Bound. Never silently narrow a
        // binary64 property while authenticating its original input columns.
        if !same(f64::from(l), lo) || !same(f64::from(u), hi) {
            return None;
        }
        input.push(Bound::new(l, u));
    }
    Some(input)
}

fn expected_parts(
    cuts: &HullRootMipCuts<'_>,
    e: &HullMipEncoding<'_>,
    poll: &mut dyn FnMut() -> bool,
) -> Option<MipParts> {
    let input = source_matches(cuts, e, poll)?;
    let mut encoder =
        crate::encode_feedforward(e.weights, e.biases, e.dimensions, &input, e.hidden_bounds)
            .ok()?;
    let row = if e.reverse {
        encoder
            .constrain_output_geq_row(e.first_output, e.second_output)
            .ok()?
    } else {
        encoder
            .constrain_output_leq_row(e.first_output, e.second_output)
            .ok()?
    };
    if row != e.unsafe_row || !poll() {
        return None;
    }
    Some(encoder.into_parts())
}

fn model_matches(
    actual: &MipParts,
    expected: &MipParts,
    unsafe_row: Row,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    if actual.num_cols != expected.num_cols
        || actual.num_cols != actual.problem.num_cols()
        || actual.input_vars != expected.input_vars
        || actual.output_vars != expected.output_vars
        || actual.problem.num_rows() != expected.problem.num_rows()
        || actual.problem.margin_row().is_some_and(|r| r != unsafe_row)
    {
        return None;
    }
    for (a, b) in actual.problem.cols().iter().zip(expected.problem.cols()) {
        if !poll()
            || !same(a.lb, b.lb)
            || !same(a.ub, b.ub)
            || !same(a.obj, b.obj)
            || (a.integer && !b.integer)
        {
            return None;
        }
    }
    for (a, b) in actual.problem.rows().iter().zip(expected.problem.rows()) {
        if !poll()
            || !same(a.lb, b.lb)
            || !same(a.ub, b.ub)
            || a.coeffs.len() != b.coeffs.len()
            || a.coeffs
                .iter()
                .zip(&b.coeffs)
                .any(|((ai, av), (bi, bv))| ai != bi || !same(*av, *bv))
        {
            return None;
        }
    }
    // Convex-binary relaxation keeps all columns/rows but filters branching
    // metadata. Authenticate THAT actual flag pattern, not the original114
    // binary count or the old mapper's all-integer precondition.
    let mut vars = Vec::new();
    let mut widths = Vec::new();
    for (&var, &width) in expected.binary_vars.iter().zip(&expected.binary_widths) {
        if actual.problem.cols().get(var.0)?.integer {
            vars.push(var);
            widths.push(width);
        }
    }
    if vars != actual.binary_vars || !same_slice(&widths, &actual.binary_widths) {
        return None;
    }
    Some(())
}

pub(super) fn head_columns(e: &HullMipEncoding<'_>, expected: &MipParts) -> Option<Vec<IrCol>> {
    let [d, h, _] = e.dimensions else { return None };
    let mut next = d.checked_add(*h)?;
    let mut binary = 0usize;
    let mut heads = Vec::with_capacity(*h);
    for (j, b) in e.hidden_bounds[0].iter().enumerate() {
        if b.lower() >= 0.0 {
            heads.push(IrCol(d.checked_add(j)?));
        } else if b.upper() <= 0.0 {
            heads.push(IrCol(next));
            next = next.checked_add(1)?;
        } else {
            heads.push(IrCol(next));
            if expected.binary_vars.get(binary) != Some(&IrCol(next.checked_add(1)?)) {
                return None;
            }
            binary = binary.checked_add(1)?;
            next = next.checked_add(2)?;
        }
    }
    if expected.output_vars.first() != Some(&IrCol(next)) || binary != expected.binary_vars.len() {
        return None;
    }
    Some(heads)
}

/// Fixed borrowed context minted only after complete original-model authentication.
/// No new map, scan or poll is performed before the optional hold is paid.
pub(super) struct SignedPhaseBinding<'a> {
    source: &'a HullMarginProblem,
    parts: &'a MipParts,
    columns: &'a [IrCol],
    bounds: &'a [Bound],
}
impl SignedPhaseBinding<'_> {
    pub(super) fn matches(&self, cuts: &HullRootMipCuts<'_>, parts: &MipParts) -> bool {
        std::ptr::eq(self.source, cuts.problem) && std::ptr::eq(self.parts, parts)
    }
    /// Original head column from the SAME fully authenticated model context.
    /// Joint row construction pays each lookup; no caller-supplied remap is trusted.
    pub(super) fn head_column(&self, head: usize) -> Option<IrCol> {
        let column = *self.columns.get(head)?;
        self.parts.problem.cols().get(column.0)?;
        Some(column)
    }
    /// Original input from the same authenticated model as the head columns.
    pub(super) fn input_column(&self, input: usize) -> Option<IrCol> {
        let column = *self.parts.input_vars.get(input)?;
        self.parts.problem.cols().get(column.0)?;
        Some(column)
    }
    /// Called only by phase preparation AFTER the same complete paid replay.
    /// model_matches already authenticated the ordered retained binary list.
    pub(super) fn column(&self, head: usize) -> Option<IrCol> {
        if *self.source.c.get(head)? >= 0. {
            return None;
        }
        let bound = self.bounds.get(head)?;
        if !(bound.lower() < 0. && bound.upper() > 0.) {
            return None;
        }
        let col = IrCol(self.columns.get(head)?.0.checked_add(1)?);
        let actual = self.parts.problem.cols().get(col.0)?;
        (actual.integer && same(actual.lb, 0.) && same(actual.ub, 1.)).then_some(col)
    }
}
/// head_columns proved the original unstable selector is headcol+1. The full
/// model comparison proved actual retained integer metadata. Only borrow here.
fn phase_binding<'a>(
    cuts: &'a HullRootMipCuts<'_>,
    parts: &'a MipParts,
    columns: &'a [IrCol],
    bounds: &'a [Bound],
) -> SignedPhaseBinding<'a> {
    SignedPhaseBinding {
        source: cuts.problem,
        parts,
        columns,
        bounds,
    }
}

#[cfg(test)]
pub(super) fn with_phase_binding_test<R>(
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    encoding: &HullMipEncoding<'_>,
    use_binding: impl FnOnce(&SignedPhaseBinding<'_>) -> R,
) -> Option<R> {
    with_phase_binding_test_polled(cuts, parts, encoding, &mut || true, &mut || {}, use_binding)
}
#[cfg(test)]
pub(super) fn with_phase_binding_test_polled<R>(
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    encoding: &HullMipEncoding<'_>,
    poll: &mut dyn FnMut() -> bool,
    authenticated: &mut dyn FnMut(),
    use_binding: impl FnOnce(&SignedPhaseBinding<'_>) -> R,
) -> Option<R> {
    let expected = expected_parts(cuts, encoding, poll)?;
    model_matches(parts, &expected, encoding.unsafe_row, poll)?;
    let columns = head_columns(encoding, &expected)?;
    authenticated();
    let binding = phase_binding(cuts, parts, &columns, &encoding.hidden_bounds[0]);
    Some(use_binding(&binding))
}

pub(super) fn prepare_rows(
    cuts: &HullRootMipCuts<'_>,
    parts: &MipParts,
    encoding: &HullMipEncoding<'_>,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<RowSpec>> {
    if !cuts.compact_mode && cuts.compact.is_some() {
        return None;
    }
    let expected = expected_parts(cuts, encoding, poll)?;
    model_matches(parts, &expected, encoding.unsafe_row, poll)?;
    let columns = head_columns(encoding, &expected)?;
    let p = cuts.problem;
    let negative: Vec<usize> =
        p.c.iter()
            .enumerate()
            .filter_map(|(j, c)| (*c < 0.0).then_some(j))
            .collect();
    if negative.len() != cuts.planes.len() {
        return None;
    }
    let mut rows = Vec::with_capacity(cuts.planes.len());
    for (plane, expected_head) in cuts.planes.iter().zip(negative) {
        if !live(deadline)
            || !poll()
            || plane.head != expected_head
            || plane.g.len() != p.input_dim
            || plane.g.iter().any(|v| !v.is_finite())
            || !same_slice(
                &plane.g,
                &cuts.multipliers.g[plane.head * p.input_dim..(plane.head + 1) * p.input_dim],
            )
        {
            return None;
        }
        let upper = majorant_offset_upper(
            p.w_row(plane.head),
            p.b1[plane.head],
            &plane.g,
            &p.lower,
            &p.upper,
        )?;
        if !same(upper, plane.upper) {
            return None;
        }
        let mut coeffs = Vec::with_capacity(p.input_dim + 1);
        coeffs.push((columns.get(plane.head)?.0, 1.0));
        for (i, &g) in plane.g.iter().enumerate() {
            if !poll() {
                return None;
            }
            if g != 0.0 {
                coeffs.push((parts.input_vars.get(i)?.0, -g));
            }
        }
        rows.push(RowSpec {
            lb: f64::NEG_INFINITY,
            ub: upper,
            coeffs,
        });
    }
    if cuts.compact_mode {
        rows.extend(compact::prepare_rows(
            cuts, parts, &columns, deadline, poll,
        )?);
    } else {
        rows.extend(joint::prepare_rows(cuts, parts, &columns, deadline, poll)?);
    }
    if let Some(global) = cuts.global_sdp.as_ref() {
        rows.extend(global.prepare_rows(cuts, parts, &columns, deadline, poll)?);
    }
    if let Some(signed) = cuts.signed.as_ref() {
        if signed.phase_admitted() || signed.portfolio_admitted() {
            let phase_binding = phase_binding(cuts, parts, &columns, &encoding.hidden_bounds[0]);
            let extra =
                signed.prepare_rows_bound(cuts, parts, &columns, &phase_binding, deadline, poll)?;
            if signed.phase_all_mode() {
                rows.reserve_exact(extra.len());
            }
            rows.extend(extra);
        } else {
            rows.extend(signed.prepare_rows(cuts, parts, &columns, deadline, poll)?);
        }
    }
    (live(deadline) && poll()).then_some(rows)
}

/// Count actual prepared input terms after excluding original heads and phase selectors.
pub(super) fn input_terms(cuts: &HullRootMipCuts<'_>, rows: &[RowSpec]) -> Option<usize> {
    let extra_heads = if cuts.compact_mode {
        cuts.joint_row_count().checked_mul(cuts.planes.len())?
    } else {
        2usize.checked_mul(cuts.joint.len())?
    }
    .checked_add(usize::from(cuts.global_sdp.is_some()).checked_mul(cuts.planes.len())?)?
    .checked_add(cuts.signed_head_terms())?;
    let input_terms = rows
        .iter()
        .map(|r| r.coeffs.len())
        .sum::<usize>()
        .checked_sub(cuts.planes.len().checked_add(extra_heads)?)?
        .checked_sub(cuts.signed_phase_terms())?;
    Some(input_terms)
}
