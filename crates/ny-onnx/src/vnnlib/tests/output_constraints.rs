// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{parse_vnnlib, OutputConstraint, VnnLibSpec};

#[ntest::timeout(10000)]
#[test]
fn test_output_constraint_equal_operator() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (= Y_0 Y_1))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.output_constraints.len(), 2);
    assert_eq!(spec.output_constraints[0], OutputConstraint::LessEq(0, 1));
    assert_eq!(
        spec.output_constraints[1],
        OutputConstraint::GreaterEq(0, 1)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_output_constraint_equality() {
    // Test that OutputConstraint derives PartialEq correctly
    let c1 = OutputConstraint::LessEq(0, 1);
    let c2 = OutputConstraint::LessEq(0, 1);
    let c3 = OutputConstraint::LessEq(1, 0);
    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_less_eq() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 Y_1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    assert_eq!(oc.num_constraints(), 1);
    assert_eq!(oc.output_dim(), 2);
    assert!(oc.is_conjunction); // Default conjunction (AND)

    // Y_0 - Y_1 <= 0: row = [1, -1], rhs = 0
    assert!((oc.a_matrix[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((oc.a_matrix[[0, 1]] - (-1.0)).abs() < 1e-6);
    assert!((oc.rhs[0] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_greater_eq() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (>= Y_0 Y_1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    // Y_0 >= Y_1 is converted to Y_1 - Y_0 <= 0: row = [-1, 1], rhs = 0
    assert!((oc.a_matrix[[0, 0]] - (-1.0)).abs() < 1e-6);
    assert!((oc.a_matrix[[0, 1]] - 1.0).abs() < 1e-6);
    assert!((oc.rhs[0] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_less_eq_const() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    // Y_0 <= 0.5: row = [1], rhs = 0.5
    assert!((oc.a_matrix[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((oc.rhs[0] - 0.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_greater_eq_const() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (>= Y_0 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    // Y_0 >= 0.5 is converted to -Y_0 <= -0.5: row = [-1], rhs = -0.5
    assert!((oc.a_matrix[[0, 0]] - (-1.0)).abs() < 1e-6);
    assert!((oc.rhs[0] - (-0.5)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_strict_inequalities() {
    // Strict inequalities should be relaxed to non-strict
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (< Y_0 Y_1))
(assert (> Y_0 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    assert_eq!(oc.num_constraints(), 2);

    // Y_0 < Y_1 treated as Y_0 - Y_1 <= 0
    assert!((oc.a_matrix[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((oc.a_matrix[[0, 1]] - (-1.0)).abs() < 1e-6);

    // Y_0 > 0.5 treated as -Y_0 <= -0.5
    assert!((oc.a_matrix[[1, 0]] - (-1.0)).abs() < 1e-6);
    assert!((oc.rhs[1] - (-0.5)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_acasxu_style() {
    // ACAS Xu argmax-style property: Y_i <= Y_0 for all i != 0
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
(assert (<= Y_3 Y_0))
(assert (<= Y_4 Y_0))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    assert_eq!(oc.num_constraints(), 4);
    assert_eq!(oc.output_dim(), 5);

    // All constraints have form Y_i - Y_0 <= 0
    for row in 0..4 {
        assert!((oc.a_matrix[[row, 0]] - (-1.0)).abs() < 1e-6); // -1 at index 0
        assert!((oc.rhs[row] - 0.0).abs() < 1e-6);
    }
    // Check specific indices: row 0 is Y_1 <= Y_0, so +1 at index 1
    assert!((oc.a_matrix[[0, 1]] - 1.0).abs() < 1e-6);
    assert!((oc.a_matrix[[1, 2]] - 1.0).abs() < 1e-6);
    assert!((oc.a_matrix[[2, 3]] - 1.0).abs() < 1e-6);
    assert!((oc.a_matrix[[3, 4]] - 1.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_disjunction() {
    // OR constraints should set is_conjunction=false
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (<= Y_0 0.0) (<= Y_1 0.0)))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert!(spec.is_disjunction); // VNN-LIB marks as disjunction

    let oc = spec.to_output_constraints().unwrap();
    assert!(!oc.is_conjunction); // INVPROP gets the inverse: NOT conjunction
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_multi_constraint_disjunction_clauses() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (and (<= Y_0 0.0) (<= Y_1 0.1))
        (and (<= Y_0 0.2) (<= Y_1 0.3))))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert!(spec.has_multi_constraint_disjunction());

    let oc = spec.to_output_constraints().unwrap();
    assert!(!oc.is_conjunction);
    let clause_indices = oc.clause_indices.as_ref().expect("clause indices");
    assert_eq!(clause_indices.len(), 2);
    assert!(clause_indices.iter().all(|c| c.len() == 2));

    let satisfies_first = ndarray::arr1(&[0.0, 0.05]);
    let satisfies_second = ndarray::arr1(&[0.1, 0.3]);
    let satisfies_none = ndarray::arr1(&[0.25, 0.2]);

    assert!(oc.is_satisfied(&satisfies_first));
    assert!(oc.is_satisfied(&satisfies_second));
    assert!(!oc.is_satisfied(&satisfies_none));
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_is_satisfied() {
    // Verify that OutputConstraints.is_satisfied matches VnnLibSpec semantics
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 Y_1))
(assert (<= Y_0 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    // Test cases that should satisfy the constraints
    let satisfied1 = ndarray::arr1(&[0.3, 0.5]); // Y_0=0.3, Y_1=0.5: both constraints met
    let satisfied2 = ndarray::arr1(&[0.5, 0.5]); // Y_0=0.5, Y_1=0.5: both constraints met (boundary)

    // Test case that should NOT satisfy
    let not_satisfied = ndarray::arr1(&[0.6, 0.5]); // Y_0=0.6: Y_0 > Y_1, Y_0 > 0.5

    assert!(oc.is_satisfied(&satisfied1));
    assert!(oc.is_satisfied(&satisfied2));
    assert!(!oc.is_satisfied(&not_satisfied));
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_rejects_zero_outputs() {
    let spec = VnnLibSpec::new(); // num_outputs = 0
    let err = spec.to_output_constraints().unwrap_err();
    match err {
        ny_core::NyError::InvalidSpec(msg) => {
            assert!(
                msg.contains("num_outputs is 0"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_empty_constraints() {
    // Spec with outputs but no constraints
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    let oc = spec.to_output_constraints().unwrap();

    assert_eq!(oc.num_constraints(), 0);
    assert_eq!(oc.output_dim(), 1);
}
