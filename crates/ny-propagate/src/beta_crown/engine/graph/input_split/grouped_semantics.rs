// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Disjunctive (OR-of-AND) domain helpers for grouped input-split BaB (#3740).
//!
//! Extracted from `disjunctive_multi_clause.rs` (Packet B1) to keep the main
//! verifier loop under the 500-line guard. These helpers mirror the grouped
//! reduction in alpha-beta-CROWN's `stop_criterion_general`
//! (`auto_LiRPA/utils.py:115-137`).

/// Validate the packed row layout shared by grouped disjunctive consumers.
///
/// Empty properties/clauses, zero-width clauses, mismatched threshold counts,
/// overflowing clause totals, and trailing/uncovered rows are all malformed.
/// Keeping this checked and allocation-free lets public fast paths reject an
/// invalid layout before any verdict bookkeeping or expensive bound collection.
pub(crate) fn valid_disjunctive_layout(
    row_count: usize,
    threshold_count: usize,
    clause_sizes: &[usize],
) -> bool {
    if row_count == 0 || threshold_count != row_count || clause_sizes.is_empty() {
        return false;
    }
    let total = clause_sizes.iter().try_fold(0usize, |acc, &size| {
        if size == 0 {
            None
        } else {
            acc.checked_add(size)
        }
    });
    total == Some(row_count)
}

/// Verdict authority for one sign-normalized objective interval.
///
/// Lower-bound verification needs a finite certified lower endpoint, a finite
/// threshold, and a non-malformed interval.  A `+inf` upper endpoint is a valid
/// one-sided enclosure and is canonical for tail/recheck paths that compute
/// only the proof-relevant lower bound; it must not erase that certificate.
/// NaN and inverted intervals still fail closed.
pub(super) fn objective_interval_verified(lower: f32, upper: f32, threshold: f32) -> bool {
    lower.is_finite()
        && !upper.is_nan()
        && threshold.is_finite()
        && lower <= upper
        && lower > threshold
}

/// Disjunctive (OR-of-AND) domain check: clause satisfied if ANY row has
/// finite `lower > threshold`; domain verified if EVERY clause satisfied.
/// Mirrors `stop_criterion_general` (`auto_LiRPA/utils.py:115-137`). Part of #3740.
pub(crate) fn disjunctive_domain_verified(
    obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> bool {
    if !valid_disjunctive_layout(obj_bounds.len(), thresholds.len(), clause_sizes) {
        return false;
    }
    let mut offset = 0;
    for &size in clause_sizes {
        let clause_satisfied = obj_bounds[offset..offset + size]
            .iter()
            .zip(&thresholds[offset..offset + size])
            .any(|(&(lower, upper), &threshold)| {
                objective_interval_verified(lower, upper, threshold)
            });
        if !clause_satisfied {
            return false;
        }
        offset += size;
    }
    true
}

/// Disjunctive (OR-of-AND) domain priority: worst clause's best row gap.
///
/// For each clause group, compute `max(lower - threshold)` over rows in that
/// clause. Then return `min` across clause groups. This reflects the grouped
/// semantics: every clause must eventually be discharged, so the priority
/// is bottlenecked by the hardest remaining clause.
///
/// Non-finite rows yield `NEG_INFINITY` gap, matching Packet A's stop helper.
///
/// Part of #3740 Packet B.
pub(crate) fn disjunctive_domain_priority(
    obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> f32 {
    if !valid_disjunctive_layout(obj_bounds.len(), thresholds.len(), clause_sizes) {
        return f32::NEG_INFINITY;
    }
    let mut offset = 0;
    let mut worst_clause = f32::INFINITY;
    for &size in clause_sizes {
        let clause_best = obj_bounds[offset..offset + size]
            .iter()
            .zip(&thresholds[offset..offset + size])
            .map(|((l, _u), &t)| {
                let gap = l - t;
                if gap.is_finite() {
                    gap
                } else {
                    f32::NEG_INFINITY
                }
            })
            .fold(f32::NEG_INFINITY, f32::max);
        worst_clause = worst_clause.min(clause_best);
        offset += size;
    }
    worst_clause
}
