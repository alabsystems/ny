// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `VnnLibSpec::shrink_input_bounds` (#4299).

use super::VnnLibSpec;

#[ntest::timeout(10000)]
#[test]
fn test_shrink_input_bounds_global_4299() {
    let mut spec = VnnLibSpec {
        num_inputs: 3,
        num_outputs: 1,
        input_bounds: vec![(0.0, 1.0), (-0.5, 0.5), (10.0, 20.0)],
        output_constraints: vec![],
        output_constraint_clauses: vec![],
        is_disjunction: false,
        version: None,
        per_clause_input_bounds: vec![],
        declared_input_bounds: vec![],
        dual_network: None,
    };

    spec.shrink_input_bounds(1e-10);

    assert!(
        (spec.input_bounds[0].0 - 1e-10).abs() < 1e-15,
        "lower[0] should increase by eps: got {}",
        spec.input_bounds[0].0
    );
    assert!(
        (spec.input_bounds[0].1 - (1.0 - 1e-10)).abs() < 1e-15,
        "upper[0] should decrease by eps: got {}",
        spec.input_bounds[0].1
    );
    assert!(
        (spec.input_bounds[1].0 - (-0.5 + 1e-10)).abs() < 1e-15,
        "lower[1] should increase by eps"
    );
    assert!(
        (spec.input_bounds[2].1 - (20.0 - 1e-10)).abs() < 1e-15,
        "upper[2] should decrease by eps"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_shrink_input_bounds_per_clause_4299() {
    use std::collections::BTreeMap;

    let mut clause_bounds = BTreeMap::new();
    clause_bounds.insert(0, (0.0, 1.0));
    clause_bounds.insert(2, (5.0, 10.0));

    let mut spec = VnnLibSpec {
        num_inputs: 3,
        num_outputs: 1,
        input_bounds: vec![(0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)],
        output_constraints: vec![],
        output_constraint_clauses: vec![],
        is_disjunction: true,
        version: None,
        per_clause_input_bounds: vec![clause_bounds],
        declared_input_bounds: vec![],
        dual_network: None,
    };

    spec.shrink_input_bounds(0.1);

    // Global bounds shrunk
    assert!(
        (spec.input_bounds[0].0 - 0.1).abs() < 1e-15,
        "global lower[0] should be 0.1"
    );
    assert!(
        (spec.input_bounds[0].1 - 0.9).abs() < 1e-15,
        "global upper[0] should be 0.9"
    );

    // Per-clause bounds also shrunk
    let clause = &spec.per_clause_input_bounds[0];
    let (lo, hi) = clause.get(&0).expect("clause bound for idx 0");
    assert!(
        (*lo - 0.1).abs() < 1e-15,
        "per-clause lower[0] should be 0.1, got {lo}"
    );
    assert!(
        (*hi - 0.9).abs() < 1e-15,
        "per-clause upper[0] should be 0.9, got {hi}"
    );
    let (lo2, hi2) = clause.get(&2).expect("clause bound for idx 2");
    assert!(
        (*lo2 - 5.1).abs() < 1e-15,
        "per-clause lower[2] should be 5.1, got {lo2}"
    );
    assert!(
        (*hi2 - 9.9).abs() < 1e-15,
        "per-clause upper[2] should be 9.9, got {hi2}"
    );
}

#[test]
#[should_panic(expected = "shrink_eps must be positive")]
fn test_shrink_input_bounds_rejects_zero_eps_4299() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = vec![(0.0, 1.0)];
    spec.shrink_input_bounds(0.0);
}

#[test]
#[should_panic(expected = "shrink_eps must be positive")]
fn test_shrink_input_bounds_rejects_negative_eps_4299() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = vec![(0.0, 1.0)];
    spec.shrink_input_bounds(-1e-10);
}

#[test]
#[should_panic(expected = "shrink_eps must be positive")]
fn test_shrink_input_bounds_rejects_nan_eps_4299() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = vec![(0.0, 1.0)];
    spec.shrink_input_bounds(f64::NAN);
}

#[test]
#[should_panic(expected = "shrink_eps must be positive")]
fn test_shrink_input_bounds_rejects_inf_eps_4299() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = vec![(0.0, 1.0)];
    spec.shrink_input_bounds(f64::INFINITY);
}
