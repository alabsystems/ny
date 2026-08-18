// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{assert_invalid_spec_contains, parse_vnnlib, OutputConstraint, VnnLibSpec};

#[ntest::timeout(10000)]
#[test]
fn test_vnnlib_spec_new() {
    let spec = VnnLibSpec::new();
    assert_eq!(spec.num_inputs, 0);
    assert_eq!(spec.num_outputs, 0);
    assert!(spec.input_bounds.is_empty());
    assert!(spec.output_constraints.is_empty());
    assert!(!spec.is_disjunction);
    assert!(spec.version.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_vnnlib_spec_default() {
    let spec = VnnLibSpec::default();
    assert_eq!(spec.num_inputs, 0);
    assert_eq!(spec.num_outputs, 0);
    assert!(spec.input_bounds.is_empty());
    assert!(spec.output_constraints.is_empty());
    assert!(!spec.is_disjunction);
    assert!(spec.version.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_is_unsafe_short_outputs_returns_false() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];

    assert!(!spec.is_unsafe(&[0.1]));
}

#[ntest::timeout(10000)]
#[test]
fn test_vnnlib_supports_logits_peel_relational() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraints = vec![
        OutputConstraint::GreaterEq(0, 1),
        OutputConstraint::LessThan(1, 0),
    ];
    assert!(spec.supports_logits_peel());
}

#[ntest::timeout(10000)]
#[test]
fn test_vnnlib_supports_logits_peel_const() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 1;
    spec.output_constraints = vec![OutputConstraint::GreaterEqConst(0, 0.5)];
    assert!(!spec.supports_logits_peel());
}

#[ntest::timeout(10000)]
#[test]
fn test_has_valid_bounds_invalid() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds.push((5.0, 1.0)); // lower > upper = invalid
    assert!(!spec.has_valid_bounds());
}

#[ntest::timeout(10000)]
#[test]
fn test_has_valid_bounds_empty() {
    let spec = VnnLibSpec::new();
    assert!(spec.has_valid_bounds()); // Empty bounds are valid
}

#[ntest::timeout(10000)]
#[test]
fn test_has_valid_bounds_equal() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds.push((1.0, 1.0)); // lower == upper is valid
    assert!(spec.has_valid_bounds());
}

#[ntest::timeout(10000)]
#[test]
fn test_describe() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))
(assert (<= Y_0 0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let desc = spec.describe();

    assert!(desc.contains("1 inputs"));
    assert!(desc.contains("1 outputs"));
    assert!(desc.contains("X_0"));
    assert!(desc.contains("Y_0"));
}

#[ntest::timeout(10000)]
#[test]
fn test_describe_all_constraint_types() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))

(assert (<= Y_0 Y_1))
(assert (>= Y_0 Y_1))
(assert (< Y_0 Y_1))
(assert (> Y_0 Y_1))
(assert (<= Y_0 0.5))
(assert (>= Y_0 -0.5))
(assert (< Y_0 1.0))
(assert (> Y_0 -1.0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let desc = spec.describe();

    // Check all constraint types appear in description
    assert!(desc.contains("Y_0 <= Y_1"));
    assert!(desc.contains("Y_0 >= Y_1"));
    assert!(desc.contains("Y_0 < Y_1"));
    assert!(desc.contains("Y_0 > Y_1"));
    assert!(desc.contains("Y_0 <= 0.5"));
    assert!(desc.contains("Y_0 >= -0.5"));
    assert!(desc.contains("Y_0 < 1.0"));
    assert!(desc.contains("Y_0 > -1.0"));
}

#[ntest::timeout(10000)]
#[test]
fn test_get_input_bounds_f32() {
    let content = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.5))
(assert (<= X_0 1.5))
(assert (>= X_1 0.0))
(assert (<= X_1 2.0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let (lower, upper) = spec.split_input_bounds_f32();

    assert_eq!(lower.len(), 2);
    assert_eq!(upper.len(), 2);
    assert!((lower[0] - (-1.5f32)).abs() < 1e-6);
    assert!((upper[0] - 1.5f32).abs() < 1e-6);
    assert!((lower[1] - 0.0f32).abs() < 1e-6);
    assert!((upper[1] - 2.0f32).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_input_bounds_f64() {
    let content = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.5))
(assert (<= X_0 1.5))
(assert (>= X_1 0.25))
(assert (<= X_1 2.75))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let (lower, upper) = spec.split_input_bounds();

    assert_eq!(lower.len(), 2);
    assert_eq!(upper.len(), 2);
    assert!((lower[0] - (-1.5)).abs() < 1e-10);
    assert!((upper[0] - 1.5).abs() < 1e-10);
    assert!((lower[1] - 0.25).abs() < 1e-10);
    assert!((upper[1] - 2.75).abs() < 1e-10);
}

// --- validate_output_indices tests (#1886) ---

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_valid_relational() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 3;
    spec.output_constraints = vec![
        OutputConstraint::LessEq(0, 1),
        OutputConstraint::GreaterEq(2, 0),
    ];
    assert!(spec.validate_output_indices().is_ok());
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_valid_const() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraints = vec![
        OutputConstraint::LessEqConst(0, 1.0),
        OutputConstraint::GreaterEqConst(1, -1.0),
    ];
    assert!(spec.validate_output_indices().is_ok());
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_oob_relational_lhs() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraints = vec![OutputConstraint::LessEq(5, 0)];
    let err = spec.validate_output_indices().unwrap_err();
    assert!(
        err.to_string().contains("Y_5"),
        "Error should mention Y_5, got: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_oob_relational_rhs() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 3;
    spec.output_constraints = vec![OutputConstraint::GreaterEq(1, 7)];
    let err = spec.validate_output_indices().unwrap_err();
    assert!(
        err.to_string().contains("Y_7"),
        "Error should mention Y_7, got: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_oob_const() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 1;
    spec.output_constraints = vec![OutputConstraint::LessEqConst(1, 0.5)];
    let err = spec.validate_output_indices().unwrap_err();
    assert!(
        err.to_string().contains("Y_1"),
        "Error should mention Y_1, got: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_oob_in_clauses() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraint_clauses = vec![
        vec![OutputConstraint::LessEq(0, 1)],       // valid
        vec![OutputConstraint::GreaterThan(0, 10)], // invalid: Y_10
    ];
    let err = spec.validate_output_indices().unwrap_err();
    assert!(
        err.to_string().contains("Y_10"),
        "Error should mention Y_10, got: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_zero_outputs() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 0;
    spec.output_constraints = vec![OutputConstraint::LessEqConst(0, 1.0)];
    let err = spec.validate_output_indices().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Y_0"),
        "Error should mention Y_0, got: {}",
        msg
    );
    assert!(
        msg.contains("no outputs declared"),
        "Error should say 'no outputs declared' when num_outputs=0, got: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_output_indices_empty_constraints_ok() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 0;
    // No constraints = nothing to validate
    assert!(spec.validate_output_indices().is_ok());
}

#[ntest::timeout(10000)]
#[test]
fn test_max_output_index_relational() {
    assert_eq!(OutputConstraint::LessEq(3, 7).max_output_index(), 7);
    assert_eq!(OutputConstraint::GreaterEq(5, 2).max_output_index(), 5);
    assert_eq!(OutputConstraint::LessThan(0, 0).max_output_index(), 0);
    assert_eq!(OutputConstraint::GreaterThan(4, 4).max_output_index(), 4);
}

#[ntest::timeout(10000)]
#[test]
fn test_max_output_index_const() {
    assert_eq!(OutputConstraint::LessEqConst(3, 1.0).max_output_index(), 3);
    assert_eq!(
        OutputConstraint::GreaterEqConst(0, -1.0).max_output_index(),
        0
    );
    assert_eq!(
        OutputConstraint::LessThanConst(5, 0.0).max_output_index(),
        5
    );
    assert_eq!(
        OutputConstraint::GreaterThanConst(2, 2.0).max_output_index(),
        2
    );
}

/// Regression test for #2658: get_input_bounds_f32 must use directed rounding
/// so the f32 region is a superset of the f64 region.
///
/// Uses 0.1 as a test value because 0.1_f64 != 0.1_f32 (neither is exactly
/// representable, and they round to different f64/f32 values).
#[ntest::timeout(10000)]
#[test]
fn test_get_input_bounds_f32_directed_rounding_2658() {
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 1;
    spec.num_outputs = 1;
    // 0.1_f64 is not exactly representable in f32.
    // f64: 0.1000000000000000055511151231257827021181583404541015625
    // f32 (round-to-nearest): 0.100000001490116119384765625
    // The f64 value is *less* than the f32 representation.
    spec.input_bounds.push((0.1_f64, 0.1_f64));

    let (lower, upper) = spec.split_input_bounds_f32();

    let val_f32_plain = 0.1_f64 as f32;
    // Lower must be <= the f64 value (rounded toward -inf).
    assert!(
        lower[0] <= val_f32_plain,
        "lower bound {:.20} should be <= plain cast {:.20} (#2658 directed rounding)",
        lower[0],
        val_f32_plain
    );
    // Upper must be >= the f64 value (rounded toward +inf).
    assert!(
        upper[0] >= val_f32_plain,
        "upper bound {:.20} should be >= plain cast {:.20} (#2658 directed rounding)",
        upper[0],
        val_f32_plain
    );
    // The f32 interval must contain the original f64 value.
    // Since f64 0.1 < f32 0.1, lower must be strictly less than plain cast.
    assert!(
        lower[0] < val_f32_plain,
        "lower bound should be strictly less than plain cast for 0.1 (not exactly representable)"
    );
}

/// Regression test for #2658: constraint rhs values must use directed rounding
/// so the unsafe region is widened (sound overapproximation).
#[ntest::timeout(10000)]
#[test]
fn test_to_output_constraints_directed_rounding_2658() {
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 1;
    spec.num_outputs = 2;
    spec.input_bounds.push((-1.0, 1.0));
    // Use a value not exactly representable in f32 as the threshold.
    spec.output_constraints = vec![
        OutputConstraint::LessEqConst(0, 0.1_f64),    // Y_0 <= 0.1
        OutputConstraint::GreaterEqConst(1, 0.1_f64), // Y_1 >= 0.1
    ];

    let constraints = spec.to_output_constraints().unwrap();

    let plain_cast = 0.1_f64 as f32;

    // For LessEqConst, plain binary32 0.1 already lies above the f64 value, so
    // the tight directed upper endpoint is the plain cast (not one extra ULP).
    assert!(
        f64::from(constraints.rhs[0]) >= 0.1_f64,
        "LessEqConst rhs {:.20} must contain f64 0.1 (#2658 directed rounding)",
        constraints.rhs[0],
    );
    assert_eq!(constraints.rhs[0], plain_cast);

    // For GreaterEqConst, the plain negated cast lies below -0.1 and must move
    // upward to contain the original f64 rhs.
    let neg_plain = -plain_cast;
    assert!(
        constraints.rhs[1] > neg_plain,
        "GreaterEqConst rhs {:.20} should be > negated plain cast {:.20} (#2658 directed rounding)",
        constraints.rhs[1],
        neg_plain
    );
}

/// A finite f64 VNN-LIB threshold that is outside binary32 range must never be
/// replaced by a finite sentinel. That would shrink the candidate violation
/// region and could let assume-violation reasoning prove the wrong region empty.
#[ntest::timeout(10000)]
#[test]
fn test_output_constraint_rhs_overflow_widens_instead_of_clamping() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 1e100))
(assert (>= Y_0 -1e100))
"#;
    let spec = parse_vnnlib(content).expect("finite f64 thresholds parse");
    let constraints = spec
        .to_output_constraints()
        .expect("overflow is represented by conservative widening");

    assert_eq!(constraints.rhs.len(), 2);
    assert!(constraints
        .rhs
        .iter()
        .all(|rhs| rhs.is_infinite() && *rhs > 0.0));
    assert!(
        constraints.is_satisfied(&ndarray::arr1(&[1.0e20_f32])),
        "the f32 lowering must contain an output admitted by the original broad region"
    );
}

/// Negative binary64 overflow has a finite, sign-correct binary32 upper
/// endpoint. Keeping it avoids throwing away the entire constraint while still
/// widening the original region.
#[ntest::timeout(10000)]
#[test]
fn test_output_constraint_negative_rhs_overflow_uses_tight_upper_endpoint() {
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 1;
    spec.num_outputs = 2;
    spec.input_bounds.push((0.0, 1.0));
    spec.output_constraints = vec![
        OutputConstraint::LessEqConst(0, -1.0e100_f64),
        OutputConstraint::GreaterEqConst(1, 1.0e100_f64),
    ];

    let constraints = spec
        .to_output_constraints()
        .expect("negative overflow has a conservative binary32 endpoint");
    assert_eq!(constraints.rhs.as_slice().unwrap(), &[f32::MIN, f32::MIN]);
    assert!(f64::from(constraints.rhs[0]) >= -1.0e100_f64);
    assert!(f64::from(constraints.rhs[1]) >= -1.0e100_f64);
}

#[ntest::timeout(10000)]
#[test]
fn test_programmatic_output_constraint_index_fails_before_matrix_indexing() {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 1;
    spec.output_constraints = vec![OutputConstraint::LessEqConst(1, 0.0)];

    let error = spec
        .to_output_constraints()
        .expect_err("out-of-range programmatic constraint must fail closed");
    let message = error.to_string();
    assert!(message.contains("Y_1"), "unexpected error: {message}");
}

/// Regression test for #2800: contradictory VNN-LIB assertions (X_0 >= 1 AND X_0 <= 0)
/// must be rejected at the VNN-LIB boundary with the offending variable named.
#[ntest::timeout(10000)]
#[test]
fn test_contradictory_vnnlib_bounds_detected_2800() {
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 1.0))
(assert (<= X_0 0.0))
(assert (<= Y_0 0.0))
";
    let err = parse_vnnlib(vnnlib_content).expect_err("contradictory bounds must be rejected");
    assert_invalid_spec_contains(err, "Input variable X_0 has an invalid bound");
}

/// Regression test for #2800: unconstrained VNN-LIB input produces infinite bounds.
/// `Bound::try_new` (finite-only) would reject these — callers must use
/// `try_new_allow_infinite` or validate before conversion.
#[ntest::timeout(10000)]
#[test]
fn test_unconstrained_vnnlib_input_produces_infinite_bounds_2800() {
    // X_0 has only an upper bound; X_1 has only a lower bound; X_2 is fully unconstrained.
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= Y_0 0.0))
";
    let spec = parse_vnnlib(vnnlib_content).expect("parse should succeed");
    let (lower, upper) = spec.split_input_bounds_f32();
    assert_eq!(
        lower[0],
        f32::NEG_INFINITY,
        "X_0 lower should be -inf (unconstrained lower)"
    );
    assert_eq!(
        upper[1],
        f32::INFINITY,
        "X_1 upper should be +inf (unconstrained upper)"
    );
    assert_eq!(
        lower[2],
        f32::NEG_INFINITY,
        "X_2 lower should be -inf (fully unconstrained)"
    );
    assert_eq!(
        upper[2],
        f32::INFINITY,
        "X_2 upper should be +inf (fully unconstrained)"
    );
}

/// Regression test for #2813: NaN constraints must propagate, not be silently absorbed.
///
/// IEEE 754 f64::max(NaN, x) returns x, which would silently drop a NaN bound
/// when intersecting multiple constraints on the same variable. The parser now
/// propagates that NaN into the stored bound and rejects it at validation.
#[ntest::timeout(10000)]
#[test]
fn test_nan_constraint_propagates_not_absorbed_2813() {
    // Two constraints on X_0: first sets lower=0.5, second asserts X_0 >= NaN.
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.5))
(assert (>= X_0 NaN))
(assert (<= X_0 1.0))
(assert (<= Y_0 0.0))
";
    let err = parse_vnnlib(vnnlib_content).expect_err("NaN lower bound must be rejected");
    assert_invalid_spec_contains(err, "Input variable X_0 has an invalid (NaN) bound");
}

/// Regression test for #2813: NaN in upper bound constraints must be rejected.
#[ntest::timeout(10000)]
#[test]
fn test_nan_upper_bound_propagates_2813() {
    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= X_0 NaN))
(assert (<= Y_0 0.0))
";
    let err = parse_vnnlib(vnnlib_content).expect_err("NaN upper bound must be rejected");
    assert_invalid_spec_contains(err, "Input variable X_0 has an invalid (NaN) bound");
}

/// ACAS Xu prop_6 shape end-to-end: a top-level OR whose disjuncts each carry
/// their OWN input box (no output atoms), plus a separate output disjunction.
/// The parser must (a) NOT intersect the disjunct boxes into the global bounds
/// (the boxes are disjoint — intersection is empty and used to fail
/// `validate_input_bounds` with "lower > upper", degrading the instance to an
/// instant sound unknown), and (b) capture each disjunct box as per-clause
/// input bounds so the boxed-clause disjunctive lane can case-split.
#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_input_box_disjunction_case_split_prop6_shape() {
    let content = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
(assert (or
    (and (<= X_1 0.5) (>= X_1 0.1))
    (and (<= X_1 -0.1) (>= X_1 -0.5))
))
(assert (or
    (and (<= Y_1 Y_0))
    (and (<= Y_0 0.0))
))
";
    let spec = parse_vnnlib(content).expect("prop_6-shaped spec must parse");
    // Global bounds: X_0 from the plain asserts; X_1 the HULL of the two
    // disjunct boxes (never their empty intersection).
    assert_eq!(spec.input_bounds[0], (-1.0, 1.0));
    assert_eq!(spec.input_bounds[1], (-0.5, 0.5));
    // 2 boxes x 2 output clauses = 4 clauses, each with its own X_1 box.
    assert!(spec.is_disjunction);
    assert_eq!(spec.output_constraint_clauses.len(), 4);
    assert_eq!(spec.per_clause_input_bounds.len(), 4);
    assert!(spec.has_boxed_clause_disjunction());
    let boxes: Vec<(f64, f64)> = spec
        .per_clause_input_bounds
        .iter()
        .map(|b| *b.get(&1).expect("every clause carries its X_1 box"))
        .collect();
    assert!(boxes.contains(&(0.1, 0.5)));
    assert!(boxes.contains(&(-0.5, -0.1)));
}

/// `declared_input_bounds` retains the TOP-LEVEL asserts EXACTLY as written,
/// captured BEFORE the per-clause union widening of `input_bounds`: a declared
/// global assert constrains EVERY clause, so witness gates must be able to
/// enforce it even when clause boxes range wider (the widened `input_bounds`
/// discards the declared value).
#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_declared_input_bounds_survive_clause_union_widening() {
    let content = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
(assert (<= X_1 0.3))
(assert (>= X_1 -0.3))
(assert (or
    (and (<= X_1 0.5) (>= X_1 0.1) (<= Y_0 0.0))
    (and (<= X_1 -0.1) (>= X_1 -0.5) (>= Y_0 1.0))
))
";
    let spec = parse_vnnlib(content).expect("widened-clause spec must parse");
    // The clause boxes exceed the declared X_1 assert, so the verification
    // domain is widened to their union hull...
    assert_eq!(spec.input_bounds[1], (-0.5, 0.5));
    assert_eq!(spec.per_clause_input_bounds.len(), 2);
    // ...but the declared top-level values survive un-widened.
    assert_eq!(spec.declared_input_bounds[0], (-1.0, 1.0));
    assert_eq!(spec.declared_input_bounds[1], (-0.3, 0.3));

    // A spec without per-clause boxes: declared == global (both un-widened),
    // and an unconstrained input stays unbounded in BOTH views.
    let plain = parse_vnnlib(
        "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
(assert (<= Y_0 0.0))
",
    )
    .expect("plain spec must parse");
    assert_eq!(plain.declared_input_bounds, plain.input_bounds);
    assert_eq!(
        plain.declared_input_bounds[1],
        (f64::NEG_INFINITY, f64::INFINITY)
    );
}
