// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::parse_vnnlib;

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (<= Y_0 Y_1))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 <= Y_1 is satisfied (unsafe region)
    assert!(spec.is_unsafe(&[0.5, 1.0]));

    // Y_0 > Y_1 is not satisfied (safe)
    assert!(!spec.is_unsafe(&[1.5, 1.0]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_with_no_output_constraints() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))
"#;

    let spec = parse_vnnlib(content).unwrap();

    assert!(spec.output_constraints.is_empty());
    assert!(spec.output_constraint_clauses.is_empty());
    assert!(!spec.is_unsafe(&[0.25]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_greater_eq_constraint() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (>= Y_0 Y_1))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 >= Y_1 is satisfied (unsafe)
    assert!(spec.is_unsafe(&[2.0, 1.0]));

    // Y_0 >= Y_1 not satisfied (safe)
    assert!(!spec.is_unsafe(&[0.5, 1.0]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_less_than_constraint() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (< Y_0 Y_1))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 < Y_1 is satisfied (unsafe)
    assert!(spec.is_unsafe(&[0.5, 1.0]));

    // Y_0 < Y_1 not satisfied when equal (safe)
    assert!(!spec.is_unsafe(&[1.0, 1.0]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_greater_than_constraint() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (> Y_0 Y_1))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 > Y_1 is satisfied (unsafe)
    assert!(spec.is_unsafe(&[1.5, 1.0]));

    // Y_0 > Y_1 not satisfied when equal (safe)
    assert!(!spec.is_unsafe(&[1.0, 1.0]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_less_eq_const() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (<= Y_0 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 <= 0.5 is satisfied
    assert!(spec.is_unsafe(&[0.5]));
    assert!(spec.is_unsafe(&[0.3]));

    // Y_0 <= 0.5 not satisfied
    assert!(!spec.is_unsafe(&[0.6]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_greater_eq_const() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (>= Y_0 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 >= 0.5 is satisfied
    assert!(spec.is_unsafe(&[0.5]));
    assert!(spec.is_unsafe(&[0.7]));

    // Y_0 >= 0.5 not satisfied
    assert!(!spec.is_unsafe(&[0.4]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_less_than_const() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (< Y_0 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 < 0.5 is satisfied
    assert!(spec.is_unsafe(&[0.4]));

    // Y_0 < 0.5 not satisfied (equal)
    assert!(!spec.is_unsafe(&[0.5]));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_greater_than_const() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (> Y_0 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();

    // Y_0 > 0.5 is satisfied
    assert!(spec.is_unsafe(&[0.6]));

    // Y_0 > 0.5 not satisfied (equal)
    assert!(!spec.is_unsafe(&[0.5]));
}
