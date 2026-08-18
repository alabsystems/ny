// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Parity tests for `disjunctive_domain_verified` against the reference
//! `stop_criterion_general` from alpha-beta-CROWN
//! (`auto_LiRPA/utils.py:115-137`). Part of #3740.

use super::*;

/// Test-local mirror of `stop_criterion_general` from alpha-beta-CROWN.
/// Groups rows by OR clause, marks a clause verified when any row in that group
/// has `lower > threshold`, then requires ALL clauses to pass.
fn reference_stop_criterion_general(
    lowers: &[f32],
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> bool {
    let mut offset = 0;
    for &size in clause_sizes {
        let clause_verified = lowers[offset..offset + size]
            .iter()
            .zip(&thresholds[offset..offset + size])
            .any(|(&l, &t)| l > t);
        if !clause_verified {
            return false;
        }
        offset += size;
    }
    true
}

// --- Test 1: requires every clause ---

#[test]
fn test_disjunctive_domain_verified_requires_every_clause_3740() {
    // Two clauses: [2 rows, 2 rows]. First clause has one verified row,
    // second clause has no verified rows.
    let obj_bounds = vec![
        (-1.0_f32, 1.0), // clause 0, row 0: lower=-1.0
        (0.5, 1.0),      // clause 0, row 1: lower=0.5 (> threshold 0.0 → verified)
        (-2.0, 0.0),     // clause 1, row 0: lower=-2.0
        (-0.5, 0.0),     // clause 1, row 1: lower=-0.5
    ];
    let thresholds = vec![0.0, 0.0, 0.0, 0.0];
    let clause_sizes = vec![2, 2];

    // One satisfied clause + one unsatisfied clause → false
    assert!(
        !disjunctive_domain_verified(&obj_bounds, &thresholds, &clause_sizes),
        "should return false when not all clauses are satisfied"
    );

    // Now make second clause also satisfied
    let obj_bounds_both = vec![
        (-1.0_f32, 1.0), // clause 0, row 0
        (0.5, 1.0),      // clause 0, row 1: verified
        (-2.0, 0.0),     // clause 1, row 0
        (0.1, 0.5),      // clause 1, row 1: lower=0.1 > 0.0 → verified
    ];

    assert!(
        disjunctive_domain_verified(&obj_bounds_both, &thresholds, &clause_sizes),
        "should return true when all clauses are satisfied"
    );
}

// --- Test 2: ignores non-finite rows ---

#[test]
fn test_disjunctive_domain_verified_ignores_nonfinite_rows_3740() {
    // Single clause with 3 rows: NaN, Inf, and a finite verified row.
    let obj_bounds_nan = vec![
        (f32::NAN, 1.0),      // NaN lower: must not count
        (f32::INFINITY, 1.0), // Inf lower: must not count (is_finite is false)
        (-0.5, 0.0),          // finite but below threshold
    ];
    let thresholds = vec![0.0, 0.0, 0.0];
    let clause_sizes = vec![3];

    // Even though NaN and Inf are "large", they must not satisfy the clause
    assert!(
        !disjunctive_domain_verified(&obj_bounds_nan, &thresholds, &clause_sizes),
        "NaN and Inf lowers must never count as verified rows"
    );

    // Add a finite verified row to prove the clause can still pass
    let obj_bounds_with_finite = vec![
        (f32::NAN, 1.0),
        (f32::INFINITY, 1.0),
        (0.1, 0.5), // finite and > threshold=0.0
    ];

    assert!(
        disjunctive_domain_verified(&obj_bounds_with_finite, &thresholds, &clause_sizes),
        "clause should pass when at least one finite row exceeds threshold"
    );

    // NEG_INFINITY lower
    let obj_bounds_neg_inf = vec![(f32::NEG_INFINITY, 0.0)];
    let thresholds_one = vec![0.0];
    let clause_sizes_one = vec![1];

    assert!(
        !disjunctive_domain_verified(&obj_bounds_neg_inf, &thresholds_one, &clause_sizes_one),
        "NEG_INFINITY lower must not count as verified"
    );
}

#[test]
fn test_disjunctive_domain_verified_rejects_malformed_interval_authority() {
    let threshold = [0.0_f32];
    let one_clause = [1usize];
    assert!(
        disjunctive_domain_verified(&[(1.0_f32, f32::INFINITY)], &threshold, &one_clause),
        "+inf is a valid upper side of a one-sided certified-lower enclosure"
    );
    for bounds in [[(1.0_f32, f32::NAN)], [(1.0_f32, 0.5_f32)]] {
        assert!(
            !disjunctive_domain_verified(&bounds, &threshold, &one_clause),
            "a NaN or inverted interval must not close a clause"
        );
    }
    assert!(
        !disjunctive_domain_verified(&[(1.0, 2.0)], &[f32::NEG_INFINITY], &one_clause),
        "a non-finite threshold must not acquire proof authority"
    );
}

#[test]
fn test_grouped_helpers_fail_closed_on_malformed_layouts() {
    let positive = vec![(1.0_f32, 1.0)];
    let threshold = vec![0.0_f32];
    let cases: Vec<(&[(f32, f32)], &[f32], &[usize])> = vec![
        (&[], &[], &[]),
        (&positive, &threshold, &[]),
        (&positive, &[], &[1]),
        (&positive, &threshold, &[0, 1]),
        (&positive, &threshold, &[2]),
        (&positive, &threshold, &[usize::MAX, 1]),
    ];

    for (obj_bounds, thresholds, clause_sizes) in cases {
        assert!(!disjunctive_domain_verified(
            obj_bounds,
            thresholds,
            clause_sizes
        ));
        assert_eq!(
            disjunctive_domain_priority(obj_bounds, thresholds, clause_sizes),
            f32::NEG_INFINITY
        );
    }
}

// --- Test 3: matches reference stop_criterion_general ---

#[test]
fn test_disjunctive_domain_verified_matches_reference_stop_criterion_3740() {
    struct Case {
        lowers: Vec<f32>,
        thresholds: Vec<f32>,
        clause_sizes: Vec<usize>,
    }

    let cases = [
        // One-clause grouping: single clause of 3 rows, one verified
        Case {
            lowers: vec![-1.0, 0.5, -0.3],
            thresholds: vec![0.0, 0.0, 0.0],
            clause_sizes: vec![3],
        },
        // Multi-clause grouping: 3 clauses [1, 2, 1], all satisfied
        Case {
            lowers: vec![0.1, -0.5, 0.3, 0.2],
            thresholds: vec![0.0, 0.0, 0.0, 0.0],
            clause_sizes: vec![1, 2, 1],
        },
        // Multi-clause: 3 clauses [1, 2, 1], middle clause unsatisfied
        Case {
            lowers: vec![0.1, -0.5, -0.3, 0.2],
            thresholds: vec![0.0, 0.0, 0.0, 0.0],
            clause_sizes: vec![1, 2, 1],
        },
        // All unsatisfied
        Case {
            lowers: vec![-1.0, -2.0],
            thresholds: vec![0.0, 0.0],
            clause_sizes: vec![1, 1],
        },
        // Non-zero thresholds
        Case {
            lowers: vec![0.5, 1.5, 0.8],
            thresholds: vec![1.0, 1.0, 1.0],
            clause_sizes: vec![2, 1],
        },
        // Edge: single row, single clause
        Case {
            lowers: vec![0.01],
            thresholds: vec![0.0],
            clause_sizes: vec![1],
        },
        // Edge: single row, single clause, not verified
        Case {
            lowers: vec![-0.01],
            thresholds: vec![0.0],
            clause_sizes: vec![1],
        },
        // Mixed satisfied/unsatisfied: [2, 3, 1]
        Case {
            lowers: vec![-0.1, 0.2, -0.5, -0.3, 0.1, -0.8],
            thresholds: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            clause_sizes: vec![2, 3, 1],
        },
    ];

    for (i, case) in cases.iter().enumerate() {
        let obj_bounds: Vec<(f32, f32)> = case.lowers.iter().map(|&l| (l, l + 1.0)).collect();
        let production =
            disjunctive_domain_verified(&obj_bounds, &case.thresholds, &case.clause_sizes);
        let reference =
            reference_stop_criterion_general(&case.lowers, &case.thresholds, &case.clause_sizes);
        assert_eq!(
            production, reference,
            "case {i}: production={production} != reference={reference} \
             (lowers={:?}, clause_sizes={:?})",
            case.lowers, case.clause_sizes
        );
    }
}

// --- Test 4: single clause reduces to existing conjunctive helper ---

#[test]
fn test_single_clause_disjunctive_helper_matches_existing_conjunctive_helper_3740() {
    // When clause_sizes = [N], disjunctive_domain_verified should match
    // multi_obj_domain_verified since both check "any row verified".
    let test_vectors: Vec<Vec<(f32, f32)>> = vec![
        vec![(-1.0, 1.0), (0.5, 1.0), (-0.3, 0.2)],
        vec![(-1.0, 0.0), (-0.5, 0.0)],
        vec![(0.1, 0.5)],
        vec![(f32::NAN, 1.0), (-0.5, 0.0)],
        vec![(f32::INFINITY, 1.0), (-0.5, 0.0)],
        vec![(f32::NEG_INFINITY, 0.0), (0.1, 0.5)],
    ];

    for (i, obj_bounds) in test_vectors.iter().enumerate() {
        let n = obj_bounds.len();
        let thresholds = vec![0.0; n];
        let clause_sizes = vec![n];

        let disjunctive = disjunctive_domain_verified(obj_bounds, &thresholds, &clause_sizes);
        let conjunctive = multi_obj_domain_verified(obj_bounds, &thresholds);

        assert_eq!(
            disjunctive, conjunctive,
            "case {i}: single-clause disjunctive ({disjunctive}) != conjunctive ({conjunctive}) \
             for bounds {:?}",
            obj_bounds
        );
    }
}
