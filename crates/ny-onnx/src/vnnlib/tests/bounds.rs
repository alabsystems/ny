// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{assert_invalid_spec_contains, parse_vnnlib, OutputConstraint};

#[ntest::timeout(10000)]
#[test]
fn test_reversed_input_bounds() {
    // Test constant op variable form
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

; Reversed form: constant <= variable
(assert (<= -1.0 X_0))
(assert (>= 1.0 X_0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 1);
    // -1.0 <= X_0 means X_0 >= -1.0 (lower bound)
    // 1.0 >= X_0 means X_0 <= 1.0 (upper bound)
    assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_reversed_output_constraints() {
    // Test constant op output variable
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

; Reversed form: 0.5 <= Y_0 means Y_0 >= 0.5
(assert (<= 0.5 Y_0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.output_constraints.len(), 1);
    // 0.5 <= Y_0 means Y_0 >= 0.5
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::GreaterEqConst(0, c) if (c - 0.5).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_and_expression() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (and (<= Y_0 Y_1) (<= Y_1 1.0)))
"#;

    let spec = parse_vnnlib(content).unwrap();
    // AND should parse both constraints
    assert_eq!(spec.output_constraints.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_or_expression_sets_disjunction() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (<= Y_0 0.0) (<= Y_1 0.0)))
"#;

    let spec = parse_vnnlib(content).unwrap();
    // OR with output constraints should set is_disjunction
    assert!(spec.is_disjunction);
    assert_eq!(spec.output_constraints.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_partial_bounds_lower_only() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.0))
; No upper bound specified
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 1);
    assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
    assert!(spec.input_bounds[0].1.is_infinite()); // Default infinity
}

#[ntest::timeout(10000)]
#[test]
fn test_partial_bounds_upper_only() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
; No lower bound specified
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 1);
    assert!(spec.input_bounds[0].0.is_infinite()); // Default -infinity
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_multiple_bounds_same_variable_takes_tightest() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -2.0))
(assert (>= X_0 -1.0))
(assert (<= X_0 2.0))
(assert (<= X_0 1.0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    // Lower bound: max(-2.0, -1.0) = -1.0
    // Upper bound: min(2.0, 1.0) = 1.0
    assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_strict_inequality_input_bounds_fail_closed() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (< X_0 1.0))
(assert (> X_0 -1.0))
"#;

    let error = parse_vnnlib(content)
        .expect_err("an open input endpoint must not become an inclusive box boundary");
    assert!(error.to_string().contains("Strict input constraints"));
}

#[ntest::timeout(10000)]
#[test]
fn test_reversed_strict_input_bounds_fail_closed() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

; val < X_0 means X_0 > val (lower bound exclusive)
(assert (< -1.0 X_0))
; val > X_0 means X_0 < val (upper bound exclusive)
(assert (> 1.0 X_0))
"#;

    let error = parse_vnnlib(content)
        .expect_err("a reversed open endpoint must not become an inclusive box boundary");
    assert!(error.to_string().contains("Strict input constraints"));
}

#[ntest::timeout(10000)]
#[test]
fn test_reversed_strict_output_constraints() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

; 0.5 < Y_0 means Y_0 > 0.5
(assert (< 0.5 Y_0))
; 1.0 > Y_0 means Y_0 < 1.0
(assert (> 1.0 Y_0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.output_constraints.len(), 2);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::GreaterThanConst(0, c) if (c - 0.5).abs() < 1e-10
    ));
    assert!(matches!(
        spec.output_constraints[1],
        OutputConstraint::LessThanConst(0, c) if (c - 1.0).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_scientific_notation_in_bounds() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.5e-3))
(assert (<= X_0 2.0E+2))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert!((spec.input_bounds[0].0 - (-0.0015)).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 200.0).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_empty_assert() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert ())
"#;

    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Empty assert expression");
}

#[ntest::timeout(10000)]
#[test]
fn test_input_constraint_equal_operator() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (= X_0 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert!((spec.input_bounds[0].0 - 0.5).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 0.5).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_reversed_input_unknown_operator_errors() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (!= 0 X_0))
"#;

    assert!(parse_vnnlib(content).is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_nested_and_in_or() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (and (<= Y_0 0.0) (<= Y_1 0.0)) (and (>= Y_0 1.0) (>= Y_1 1.0))))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert!(spec.is_disjunction);
    assert_eq!(spec.output_constraint_clauses.len(), 2);
    assert!(spec
        .output_constraint_clauses
        .iter()
        .all(|clause| clause.len() == 2));
}

#[ntest::timeout(10000)]
#[test]
fn test_sparse_variable_indices() {
    // Test that sparse indices work (X_0, X_5, X_2)
    let content = r#"
(declare-const X_0 Real)
(declare-const X_5 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (>= X_5 5))
(assert (<= X_5 6))
(assert (>= X_2 2))
(assert (<= X_2 3))
"#;

    let spec = parse_vnnlib(content).unwrap();
    // num_inputs should be max_idx + 1 = 6
    assert_eq!(spec.num_inputs, 6);
    // Bounds for undefined indices should be (-inf, +inf)
    assert!(spec.input_bounds[1].0.is_infinite());
    assert!(spec.input_bounds[3].0.is_infinite());
    assert!(spec.input_bounds[4].0.is_infinite());
}
