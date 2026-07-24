// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Comparison utilities for NaN-safe sorting.
//!
//! # Problem (#2981 Slice 3)
//!
//! 16+ sorting sites used `partial_cmp(...).unwrap_or(Ordering::Equal)`,
//! which treats NaN as equal to any value. In BaB branching, this corrupts
//! priority queues: NaN-scored candidates sort into arbitrary positions
//! instead of being pushed to the end where they won't be selected.
//!
//! # Solution
//!
//! - [`nan_propagating_cmp`] — ascending sort / `min_by` with NaN last.
//! - [`nan_least_cmp`] — `max_by` with NaN deprioritized (NaN < everything).
//! - [`nan_last_descending_cmp`] — descending sort with NaN last (#2995).
//!
//! Together these ensure NaN-scored items are deprioritized regardless of
//! sort direction or selection method. Do NOT reverse arguments to
//! `nan_propagating_cmp` for descending sort — use `nan_last_descending_cmp`
//! instead. Do NOT use `f32::total_cmp` with `max_by` — NaN > Infinity in
//! total ordering means NaN would be selected as the maximum.

use std::cmp::Ordering;

/// Compare f32 values for sorting, placing NaN at the end (ascending order).
///
/// NaN is treated as greater than all finite values and ±Inf.
/// In ascending sorts, NaN appears last.
///
/// **Warning:** Do NOT use `nan_propagating_cmp(b, a)` for descending sort —
/// NaN will sort FIRST (opposite of intended). Use [`nan_last_descending_cmp`]
/// instead.
///
/// # Design
///
/// - `finite.cmp(finite)` → normal ordering
/// - `finite.cmp(NaN)` → Less (finite before NaN)
/// - `NaN.cmp(finite)` → Greater (NaN after finite)
/// - `NaN.cmp(NaN)` → Equal
///
/// Source: `designs/2026-02-25-error-swallowing-classification.md` Slice 3
pub(crate) fn nan_propagating_cmp(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // At least one is NaN — NaN sorts last
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater, // NaN after finite
            (false, true) => Ordering::Less,    // finite before NaN
            (false, false) => unreachable!("partial_cmp returned None but neither is NaN"),
        }
    })
}

/// Compare f32 values treating NaN as less than all finite values.
///
/// Use with `max_by` to ensure NaN values are never selected as the maximum.
/// `nan_propagating_cmp` treats NaN as greater (correct for `min_by` but wrong
/// for `max_by` where NaN would be selected). This function inverts the NaN
/// ordering so NaN always loses in `max_by`.
///
/// Complement of `nan_propagating_cmp`:
/// - `nan_propagating_cmp`: use with `min_by`, `sort_by` (ascending)
/// - `nan_least_cmp`: use with `max_by`
/// - `nan_last_descending_cmp`: use with `sort_by` (descending)
pub(crate) fn nan_least_cmp(a: &f32, b: &f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less, // NaN loses (less than finite)
        (false, true) => Ordering::Greater, // finite wins (greater than NaN)
        (false, false) => a.total_cmp(b),
    }
}

/// Compare f32 values for descending sort, placing NaN at the end.
///
/// Finite values sort in descending order (largest first). NaN values
/// always sort last, regardless of direction. This avoids the pitfall of
/// reversing `nan_propagating_cmp` which places NaN first.
///
/// Uses `f32::total_cmp` for the non-NaN branch to avoid any
/// `partial_cmp().unwrap_or(Equal)` pattern (#2588).
pub(crate) fn nan_last_descending_cmp(a: &f32, b: &f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN after finite (last)
        (false, true) => Ordering::Less,    // finite before NaN
        // Both non-NaN: reverse total_cmp for descending order.
        (false, false) => b.total_cmp(a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_propagating_cmp_finite_values() {
        assert_eq!(nan_propagating_cmp(&1.0, &2.0), Ordering::Less);
        assert_eq!(nan_propagating_cmp(&2.0, &1.0), Ordering::Greater);
        assert_eq!(nan_propagating_cmp(&1.0, &1.0), Ordering::Equal);
        assert_eq!(nan_propagating_cmp(&-1.0, &1.0), Ordering::Less);
    }

    #[test]
    fn test_nan_propagating_cmp_nan_sorts_last() {
        assert_eq!(nan_propagating_cmp(&f32::NAN, &1.0), Ordering::Greater);
        assert_eq!(nan_propagating_cmp(&1.0, &f32::NAN), Ordering::Less);
        assert_eq!(nan_propagating_cmp(&f32::NAN, &f32::NAN), Ordering::Equal);
    }

    #[test]
    fn test_nan_propagating_cmp_nan_after_infinity() {
        assert_eq!(
            nan_propagating_cmp(&f32::NAN, &f32::INFINITY),
            Ordering::Greater
        );
        assert_eq!(
            nan_propagating_cmp(&f32::INFINITY, &f32::NAN),
            Ordering::Less
        );
        assert_eq!(
            nan_propagating_cmp(&f32::NAN, &f32::NEG_INFINITY),
            Ordering::Greater
        );
    }

    #[test]
    fn test_nan_propagating_cmp_sort_ascending() {
        let mut values = [3.0, f32::NAN, 1.0, f32::NAN, 2.0];
        values.sort_by(nan_propagating_cmp);
        // NaN should be at the end
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
    }

    #[test]
    fn test_nan_last_descending_cmp_sort() {
        let mut values = [3.0, f32::NAN, 1.0, f32::NAN, 2.0];
        values.sort_by(nan_last_descending_cmp);
        // Descending with NaN at the end
        assert_eq!(values[0], 3.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 1.0);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
    }

    #[test]
    fn test_nan_last_descending_cmp_with_infinity() {
        let mut values = [1.0, f32::INFINITY, f32::NAN, f32::NEG_INFINITY, 2.0];
        values.sort_by(nan_last_descending_cmp);
        assert_eq!(values[0], f32::INFINITY);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 1.0);
        assert_eq!(values[3], f32::NEG_INFINITY);
        assert!(values[4].is_nan());
    }

    /// Acceptance criterion for #2588: BaBSR-style scored tuples with NaN scores
    /// produce deterministic sort order where NaN-scored neurons sort to the end.
    #[test]
    fn test_babsr_scored_tuples_nan_sorts_last_descending() {
        // Simulate BaBSR scored tuples: (layer_idx, neuron_idx, score)
        let mut scored: Vec<(usize, usize, f32)> = vec![
            (1, 0, 0.5),      // valid
            (1, 1, f32::NAN), // NaN-scored (corrupt CROWN coefficient)
            (2, 0, 1.2),      // valid, highest
            (1, 2, f32::NAN), // NaN-scored
            (2, 1, 0.8),      // valid
        ];
        // Sort descending by score (highest first, NaN last)
        scored.sort_by(|a, b| nan_last_descending_cmp(&a.2, &b.2));

        // Valid scores in descending order, NaN at end
        assert_eq!(scored[0], (2, 0, 1.2));
        assert_eq!(scored[1], (2, 1, 0.8));
        assert_eq!(scored[2], (1, 0, 0.5));
        // NaN-scored neurons are deprioritized — never selected for branching
        assert!(scored[3].2.is_nan());
        assert!(scored[4].2.is_nan());
    }

    /// Acceptance criterion for #2588: ascending sort with NaN intercept scores
    /// places NaN at the end so k-smallest selection picks valid neurons.
    #[test]
    fn test_intercept_scored_tuples_nan_sorts_last_ascending() {
        // Simulate intercept-only scores: (layer_idx, neuron_idx, score)
        let mut scored: Vec<(usize, usize, f32)> = vec![
            (1, 0, -0.3),     // valid (most negative = best in k-smallest)
            (1, 1, f32::NAN), // NaN
            (2, 0, 0.1),      // valid
            (2, 1, -0.7),     // valid, best
        ];
        // Sort ascending (k-smallest), NaN last
        scored.sort_by(|a, b| nan_propagating_cmp(&a.2, &b.2));

        assert_eq!(scored[0], (2, 1, -0.7));
        assert_eq!(scored[1], (1, 0, -0.3));
        assert_eq!(scored[2], (2, 0, 0.1));
        assert!(scored[3].2.is_nan());
    }

    /// Acceptance criterion for #2588: BaBSR max-finding with NaN in first
    /// candidate must still select a valid neuron (not stick on NaN).
    #[test]
    fn test_max_finding_nan_first_candidate_selects_valid() {
        // Simulate the max-finding loop pattern from select_babsr_neuron
        let candidates = [
            (1usize, 0usize, f32::NAN), // NaN first candidate
            (1, 1, 0.5),                // valid
            (2, 0, 1.2),                // valid, highest
        ];

        // This is the corrected pattern: skip NaN, then track best
        let mut best: Option<(usize, usize, f32)> = None;
        for &(layer, neuron, score) in &candidates {
            if score.is_nan() {
                continue;
            }
            if best.map_or(true, |b| score > b.2) {
                best = Some((layer, neuron, score));
            }
        }

        let best = best.expect("should find a valid candidate");
        assert_eq!(best, (2, 0, 1.2));
        assert!(!best.2.is_nan(), "selected neuron must not have NaN score");
    }

    #[test]
    fn test_nan_least_cmp_finite_values() {
        assert_eq!(nan_least_cmp(&1.0, &2.0), Ordering::Less);
        assert_eq!(nan_least_cmp(&2.0, &1.0), Ordering::Greater);
        assert_eq!(nan_least_cmp(&1.0, &1.0), Ordering::Equal);
    }

    #[test]
    fn test_nan_least_cmp_nan_is_less_than_everything() {
        assert_eq!(nan_least_cmp(&f32::NAN, &1.0), Ordering::Less);
        assert_eq!(nan_least_cmp(&1.0, &f32::NAN), Ordering::Greater);
        assert_eq!(nan_least_cmp(&f32::NAN, &f32::NAN), Ordering::Equal);
        assert_eq!(nan_least_cmp(&f32::NAN, &f32::NEG_INFINITY), Ordering::Less);
    }

    /// Verify nan_least_cmp works correctly with max_by: NaN is never selected
    /// as the maximum. This is the complement of nan_propagating_cmp for min_by.
    #[test]
    fn test_nan_least_cmp_max_by_skips_nan() {
        let values = [1.0_f32, f32::NAN, 3.0, f32::NAN, 2.0];
        let max = values.iter().max_by(|a, b| nan_least_cmp(a, b));
        assert_eq!(max, Some(&3.0));
    }

    /// Verify nan_propagating_cmp works correctly with min_by: NaN is never
    /// selected as the minimum.
    #[test]
    fn test_nan_propagating_cmp_min_by_skips_nan() {
        let values = [1.0_f32, f32::NAN, 3.0, f32::NAN, 2.0];
        let min = values.iter().min_by(|a, b| nan_propagating_cmp(a, b));
        assert_eq!(min, Some(&1.0));
    }
}
