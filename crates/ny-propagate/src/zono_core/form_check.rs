// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sampled containment checks for zonotope forms — the checker the soundness
//! tests of every zonotope user share.
//!
//! Two checks per sampled point and node:
//!
//! * BOX: the concrete value lies in the concretised interval.
//! * POINTWISE FORM: with every input symbol set to its value at the point,
//!   `e_d = (x_d - center_d) / radius_d`, the residual
//!   `|f(x) - c - sum_d G_d e_d|` is at most `sum_{other} |G_k| + r`.
//!
//! The form check is strictly stronger: F1 measured a McCormick-sign defect
//! that every box check passed and the form check caught, and the V2 mutation
//! pins (flipped DeepZ offset, dropped remainder) are caught the same way.
//! The form tolerance only absorbs the CHECKER's own binary64 evaluation
//! (relative `1e-10`); a real defect is many orders larger.

use super::concretize::concretize;
use super::program::ZonoNodeView;

/// Relative tolerance of the pointwise form check's own evaluation.
pub const FORM_CHECK_RELATIVE_TOLERANCE: f64 = 1e-10;

/// A compact snapshot of one node's form: enough for both checks at any
/// number of points without keeping the full generator matrix.
#[derive(Debug, Clone)]
pub struct FormSummary {
    /// Node name.
    pub node: String,
    /// Centre.
    pub center: Vec<f64>,
    /// `(input dim, center, radius, generator row)` per input symbol present.
    pub input_rows: Vec<(usize, f64, f64, Vec<f64>)>,
    /// `sum |G_k|` over every non-input symbol.
    pub other_abs: Vec<f64>,
    /// Remainder.
    pub rem: Vec<f64>,
    /// Concretised lower bound.
    pub lo: Vec<f64>,
    /// Concretised upper bound.
    pub hi: Vec<f64>,
}

/// Snapshot the node an observer is looking at.
#[must_use]
pub fn summarize(view: &ZonoNodeView<'_>) -> FormSummary {
    let state = view.state;
    let n = state.len();
    let mut input_rows = Vec::new();
    let mut is_input = vec![false; state.num_generators()];
    for symbol in view.input_symbols {
        if let Some(row) = state.ids().iter().position(|&id| id == symbol.id) {
            is_input[row] = true;
            input_rows.push((
                symbol.dim,
                symbol.center,
                symbol.radius,
                state.generator_row(row).to_vec(),
            ));
        }
    }
    let mut other_abs = vec![0.0f64; n];
    for (row, &input) in is_input.iter().enumerate() {
        if !input {
            for (slot, value) in other_abs.iter_mut().zip(state.generator_row(row)) {
                *slot += value.abs();
            }
        }
    }
    let (lo, hi) = concretize(state, view.abs_g);
    FormSummary {
        node: view.node.to_string(),
        center: state.center().to_vec(),
        input_rows,
        other_abs,
        rem: state.remainder().to_vec(),
        lo,
        hi,
    }
}

/// Violation counters over many points.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ContainmentReport {
    /// Point-node values checked.
    pub checked: usize,
    /// Values outside the concretised box.
    pub box_violations: usize,
    /// Values whose form residual exceeded its radius.
    pub form_violations: usize,
    /// Largest box excess seen.
    pub worst_box_excess: f64,
    /// Largest form excess seen.
    pub worst_form_excess: f64,
}

impl ContainmentReport {
    /// Merge another report into this one.
    pub fn absorb(&mut self, other: &Self) {
        self.checked += other.checked;
        self.box_violations += other.box_violations;
        self.form_violations += other.form_violations;
        self.worst_box_excess = self.worst_box_excess.max(other.worst_box_excess);
        self.worst_form_excess = self.worst_form_excess.max(other.worst_form_excess);
    }
}

/// Check one point: `input` is the flat input point, `values` the node's
/// concrete values at it.
pub fn check_point(
    summary: &FormSummary,
    input: &[f64],
    values: &[f64],
    report: &mut ContainmentReport,
) {
    let symbol_values: Vec<f64> = summary
        .input_rows
        .iter()
        .map(|(dim, center, radius, _)| (input[*dim] - center) / radius)
        .collect();
    for (j, &value) in values.iter().enumerate() {
        report.checked += 1;
        let box_excess = (summary.lo[j] - value).max(value - summary.hi[j]);
        if box_excess > 0.0 {
            report.box_violations += 1;
            report.worst_box_excess = report.worst_box_excess.max(box_excess);
        }
        let mut predicted = summary.center[j];
        let mut scale = summary.center[j].abs();
        for ((_, _, _, row), e) in summary.input_rows.iter().zip(&symbol_values) {
            predicted += row[j] * e;
            scale += row[j].abs();
        }
        let radius = summary.other_abs[j] + summary.rem[j];
        let tolerance =
            FORM_CHECK_RELATIVE_TOLERANCE * (1.0 + value.abs() + scale + summary.other_abs[j]);
        let form_excess = (value - predicted).abs() - radius - tolerance;
        if form_excess > 0.0 {
            report.form_violations += 1;
            report.worst_form_excess = report.worst_form_excess.max(form_excess);
        }
    }
}
