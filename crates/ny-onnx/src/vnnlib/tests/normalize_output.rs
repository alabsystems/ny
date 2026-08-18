// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::OutputConstraint;
use crate::vnnlib::normalize as vnn_normalize;

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_linearizes_arithmetic() {
    let content = r#"
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= (- Y_0 Y_1) 0.0))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    assert_eq!(normalized.clauses.len(), 1);
    assert_eq!(normalized.clauses[0].len(), 1);
    let constraint = &normalized.clauses[0][0];
    assert_eq!(constraint.relation, vnn_normalize::Relation::LessEq);
    assert!((constraint.expr.coeff(vnn_normalize::VarKind::Output, 0) - 1.0).abs() < 1e-6);
    assert!((constraint.expr.coeff(vnn_normalize::VarKind::Output, 1) + 1.0).abs() < 1e-6);
    assert!(constraint.expr.constant_term().abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_dnf() {
    let content = r#"
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (or (and (<= Y_0 Y_1) (<= Y_1 1.0)) (<= Y_0 0.0)))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    assert_eq!(normalized.clauses.len(), 2);
    let lengths: Vec<usize> = normalized.clauses.iter().map(|c| c.len()).collect();
    assert!(lengths.contains(&2));
    assert!(lengths.contains(&1));
    assert!(normalized.is_disjunction());
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_to_output_constraints() {
    let content = r#"
(declare-const Y_0 Real)
(assert (or (<= Y_0 0.0) (>= Y_0 1.0)))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    let (constraints, is_disjunction) = vnn_normalize::to_output_constraints(&normalized).unwrap();
    assert!(is_disjunction);
    assert_eq!(constraints.len(), 2);
    assert!(constraints
        .iter()
        .any(|c| matches!(c, OutputConstraint::LessEqConst(0, v) if *v == 0.0)));
    assert!(constraints
        .iter()
        .any(|c| matches!(c, OutputConstraint::GreaterEqConst(0, v) if *v == 1.0)));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_disjunction_with_conjunction_clause() {
    let content = r#"
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (or (and (<= Y_0 0.0) (<= Y_1 0.0)) (<= Y_0 1.0)))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    let (clauses, is_disjunction) =
        vnn_normalize::to_output_constraint_clauses(&normalized).unwrap();
    assert!(is_disjunction);
    assert_eq!(clauses.len(), 2);
    let lengths: Vec<usize> = clauses.iter().map(|c| c.len()).collect();
    assert!(lengths.contains(&2));
    assert!(lengths.contains(&1));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_disjunction_multi_mapping() {
    let content = r#"
(declare-const Y_0 Real)
(assert (or (= Y_0 0.0) (<= Y_0 1.0)))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    let (clauses, is_disjunction) =
        vnn_normalize::to_output_constraint_clauses(&normalized).unwrap();
    assert!(is_disjunction);
    assert_eq!(clauses.len(), 2);
    assert!(clauses.iter().any(|c| c.len() == 2)); // equality expands to two constraints
    assert!(clauses.iter().any(|c| c.len() == 1));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_rejects_mixed_and_or() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or (<= X_0 1.0) (<= Y_0 0.0)))
"#;
    let err = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("Mixed input-only"));
}

/// Regression test for #3192: VNNLIB disjunctive format with mixed input+output
/// constraints in `(and ...)` clauses (nn4sys lindex pattern).
/// Input bounds within `and` clauses must be accepted and extracted as per-clause
/// preconditions, not rejected.
#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_accepts_mixed_and_3192() {
    // Pattern from nn4sys lindex: each disjunct has input bounds + output constraint
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
  (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 100.0))
  (and (<= X_0 1.0) (>= X_0 0.5) (<= Y_0 200.0))
))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    // Should produce 2 disjunctive clauses, each with 1 output constraint
    assert_eq!(normalized.clauses.len(), 2);
    // Per-clause input bounds should be extracted
    assert_eq!(normalized.per_clause_input_bounds.len(), 2);
    // Clause 0: X_0 in [0.0, 0.5]
    let bounds_0 = &normalized.per_clause_input_bounds[0];
    assert!(bounds_0.contains_key(&0)); // X_0 index
                                        // Clause 1: X_0 in [0.5, 1.0]
    let bounds_1 = &normalized.per_clause_input_bounds[1];
    assert!(bounds_1.contains_key(&0));
}

/// ACAS Xu prop_6 shape: a PURE input-box disjunction assert (each disjunct
/// carries only its own input box) combined with a separate output
/// disjunction. The normalizer must distribute the output clauses over the
/// disjunct boxes (DNF), yielding per-clause input bounds — the boxed-clause
/// disjunctive shape. Historically the input-box assert was DROPPED here and
/// its atoms intersected into an empty global box (instant sound unknown).
#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_distributes_input_box_disjunction() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (or
  (and (<= X_0 0.5) (>= X_0 0.1))
  (and (<= X_0 -0.1) (>= X_0 -0.5))
))
(assert (or
  (and (<= Y_0 Y_1))
  (and (<= Y_0 0.0))
))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    // 2 boxes x 2 output clauses = 4 DNF clauses, each with 1 output constraint
    // and its OWN input box.
    assert_eq!(normalized.clauses.len(), 4);
    assert!(normalized.clauses.iter().all(|c| c.len() == 1));
    assert_eq!(normalized.per_clause_input_bounds.len(), 4);
    let boxes: Vec<(f64, f64)> = normalized
        .per_clause_input_bounds
        .iter()
        .map(|b| *b.get(&0).expect("every clause carries its X_0 box"))
        .collect();
    // Each disjunct box appears (paired with each output clause); the two
    // boxes are [0.1, 0.5] and [-0.5, -0.1].
    assert!(boxes.contains(&(0.1, 0.5)));
    assert!(boxes.contains(&(-0.5, -0.1)));
}

/// Input-box disjunctions are only captured when the spec ALSO has output
/// constraints: an input-only spec keeps the legacy empty normalization
/// (clauses with no output constraint have no supported semantics downstream).
#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_input_box_disjunction_alone_is_dropped() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
  (and (<= X_0 0.5) (>= X_0 0.1))
  (and (<= X_0 -0.1) (>= X_0 -0.5))
))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    assert!(normalized.clauses.is_empty());
    assert!(normalized.per_clause_input_bounds.is_empty());
}

/// Regression test for #3192: single mixed `and` (non-disjunctive) should also work.
#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_accepts_single_mixed_and_3192() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (and (<= X_0 1.0) (>= X_0 0.0) (<= Y_0 50.0)))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap();
    // Single conjunctive clause with 1 output constraint
    assert_eq!(normalized.clauses.len(), 1);
    assert_eq!(normalized.per_clause_input_bounds.len(), 1);
    let bounds = &normalized.per_clause_input_bounds[0];
    assert!(bounds.contains_key(&0)); // X_0 bound extracted
}

#[ntest::timeout(10000)]
#[test]
fn mixed_and_rejects_unrepresentable_input_atom_instead_of_dropping_clause() {
    let content = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (and (<= (+ X_0 X_1) 1.0) (>= Y_0 0.0)))
"#;
    let error = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect_err("a partially represented unsafe clause must fail closed");
    assert!(error
        .to_string()
        .contains("cannot be represented as one per-clause bound"));
}

#[ntest::timeout(10000)]
#[test]
fn mixed_and_retains_nested_input_conjunction() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (and
  (and (>= X_0 0.25) (<= X_0 0.75))
  (>= Y_0 0.0)
))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect("nested representable input bounds");
    assert_eq!(normalized.clauses.len(), 1);
    assert_eq!(
        normalized.per_clause_input_bounds[0].get(&0),
        Some(&(0.25, 0.75))
    );
}

#[ntest::timeout(10000)]
#[test]
fn mixed_and_rejects_strict_input_endpoint_but_preserves_strict_output() {
    let strict_input = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (and (< X_0 1.0) (> Y_0 0.0)))
"#;
    let error = vnn_normalize::normalize_output_constraints(
        strict_input,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect_err("an exclusive input endpoint must fail closed");
    assert!(error.to_string().contains("Strict input constraints"));

    let strict_output = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (and (<= X_0 1.0) (> Y_0 0.0)))
"#;
    let normalized = vnn_normalize::normalize_output_constraints(
        strict_output,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect("strict output comparisons remain supported");
    let (clauses, _) =
        vnn_normalize::to_output_constraint_clauses(&normalized).expect("convert strict output");
    assert!(matches!(
        clauses.as_slice(),
        [clause]
            if matches!(
                clause.as_slice(),
                [OutputConstraint::GreaterThanConst(0, value)] if *value == 0.0
            )
    ));
}

#[ntest::timeout(10000)]
#[test]
fn nested_strict_input_endpoint_fails_closed() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (and
  (and (>= X_0 0.25) (< X_0 0.75))
  (>= Y_0 0.0)
))
"#;
    let error = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect_err("a nested exclusive input endpoint must fail closed");
    assert!(error.to_string().contains("Strict input constraints"));
}

#[ntest::timeout(10000)]
#[test]
fn strict_input_box_disjunction_fails_closed() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
  (and (>= X_0 0.1) (< X_0 0.5))
  (and (> X_0 -0.5) (<= X_0 -0.1))
))
(assert (>= Y_0 0.0))
"#;
    let error = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect_err("strict input-box disjuncts must not become closed clause boxes");
    assert!(error.to_string().contains("Strict input constraints"));
}

#[ntest::timeout(10000)]
#[test]
fn mixed_and_rejects_partially_supported_nested_input_tree() {
    let content = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (and
  (or (<= (+ X_0 X_1) 1.0) (>= X_0 0.0))
  (>= Y_0 0.0)
))
"#;
    let error = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .expect_err("unsupported nested input logic must not disappear");
    assert!(error
        .to_string()
        .contains("Unsupported input expression in mixed input/output"));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_rejects_mixed_variables() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= (+ X_0 Y_0) 0.0))
"#;
    let err = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions::default(),
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("Constraint mixes input and output variables"));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_max_clauses() {
    let content = r#"
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (or (<= Y_0 0.0) (<= Y_1 0.0)))
"#;
    let err = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions {
            max_clauses: 1,
            max_clause_len: 16,
        },
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("DNF clause count 2 exceeds max_clauses 1"));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_output_constraints_max_clause_len() {
    let content = r#"
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (and (<= Y_0 0.0) (<= Y_1 0.0)))
"#;
    let err = vnn_normalize::normalize_output_constraints(
        content,
        vnn_normalize::NormalizeOptions {
            max_clauses: 8,
            max_clause_len: 1,
        },
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("Clause length 2 exceeds max_clause_len 1"));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_to_cnf_max_clauses() {
    let expr = vnn_normalize::BoolExpr::And(vec![
        vnn_normalize::BoolExpr::Atom(1),
        vnn_normalize::BoolExpr::Atom(2),
        vnn_normalize::BoolExpr::Atom(3),
    ]);
    let err = vnn_normalize::to_cnf(
        &expr,
        &vnn_normalize::NormalizeOptions {
            max_clauses: 2,
            max_clause_len: 8,
        },
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("CNF clause count 3 exceeds max_clauses 2"));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_to_dnf_max_clause_len() {
    let expr = vnn_normalize::BoolExpr::And(vec![
        vnn_normalize::BoolExpr::Atom(1),
        vnn_normalize::BoolExpr::Atom(2),
    ]);
    let err = vnn_normalize::to_dnf(
        &expr,
        &vnn_normalize::NormalizeOptions {
            max_clauses: 8,
            max_clause_len: 1,
        },
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("Clause length 2 exceeds max_clause_len 1"));
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_to_cnf_or_flattens() {
    let expr = vnn_normalize::BoolExpr::Or(vec![
        vnn_normalize::BoolExpr::Atom(1),
        vnn_normalize::BoolExpr::Atom(2),
    ]);
    let cnf = vnn_normalize::to_cnf(
        &expr,
        &vnn_normalize::NormalizeOptions {
            max_clauses: 8,
            max_clause_len: 8,
        },
    )
    .unwrap();
    assert_eq!(cnf, vec![vec![1, 2]]);
}

/// Reference DNF (the pre-fast-path algorithm, verbatim): every `And` child —
/// atom or not — goes through the full cross-product combine. The production
/// `to_dnf`'s atom fast path (#vnnlib-parse-dnf) must be BIT-IDENTICAL to
/// this on every input (same clauses, same order, same literal order).
#[cfg(test)]
mod dnf_reference {
    use crate::vnnlib::normalize::{BoolExpr, NormalizeOptions};
    use ny_core::{NyError, Result};

    fn combine_clauses<T: Clone>(
        left: &[Vec<T>],
        right: &[Vec<T>],
        options: &NormalizeOptions,
    ) -> Result<Vec<Vec<T>>> {
        let mut combined = Vec::new();
        for l in left {
            for r in right {
                let mut clause = Vec::with_capacity(l.len() + r.len());
                clause.extend_from_slice(l);
                clause.extend_from_slice(r);
                if clause.len() > options.max_clause_len {
                    return Err(NyError::InvalidSpec(format!(
                        "Clause length {} exceeds max_clause_len {}",
                        clause.len(),
                        options.max_clause_len
                    )));
                }
                combined.push(clause);
                if combined.len() > options.max_clauses {
                    return Err(NyError::InvalidSpec(format!(
                        "DNF/CNF clause count {} exceeds max_clauses {}",
                        combined.len(),
                        options.max_clauses
                    )));
                }
            }
        }
        Ok(combined)
    }

    pub fn to_dnf<T: Clone>(expr: &BoolExpr<T>, options: &NormalizeOptions) -> Result<Vec<Vec<T>>> {
        match expr {
            BoolExpr::Atom(atom) => Ok(vec![vec![atom.clone()]]),
            BoolExpr::Or(children) => {
                let mut clauses = Vec::new();
                for child in children {
                    let child_clauses = to_dnf(child, options)?;
                    clauses.extend(child_clauses);
                    if clauses.len() > options.max_clauses {
                        return Err(NyError::InvalidSpec(format!(
                            "DNF clause count {} exceeds max_clauses {}",
                            clauses.len(),
                            options.max_clauses
                        )));
                    }
                }
                Ok(clauses)
            }
            BoolExpr::And(children) => {
                let mut clauses = vec![Vec::new()];
                for child in children {
                    let child_clauses = to_dnf(child, options)?;
                    clauses = combine_clauses(&clauses, &child_clauses, options)?;
                }
                Ok(clauses)
            }
        }
    }
}

/// Bit-identity gate for the `to_dnf` atom fast path (#vnnlib-parse-dnf):
/// on a structured family of expressions — including the nn4sys mscn shape
/// `(or (and <many atoms>) ...)`, mixed atom/Or `And` children (fast path
/// interleaved with the general combine), nested distribution, and error
/// cases — the production `to_dnf` must agree EXACTLY with the reference
/// (pre-fast-path) algorithm: same Ok clauses in the same order, or the same
/// error string.
#[ntest::timeout(10000)]
#[test]
fn dnf_atom_fast_path_matches_reference() {
    use vnn_normalize::BoolExpr as B;

    fn atoms(range: std::ops::Range<i32>) -> Vec<B<i32>> {
        range.map(B::Atom).collect()
    }

    let exprs: Vec<B<i32>> = vec![
        // mscn shape: Or of pure-atom Ands.
        B::Or(
            (0..20)
                .map(|k| B::And(atoms(100 * k..100 * k + 37)))
                .collect(),
        ),
        // And with atoms interleaved around an Or child (fast path before,
        // general combine in the middle, fast path after).
        B::And(vec![
            B::Atom(1),
            B::Atom(2),
            B::Or(vec![
                B::And(atoms(10..14)),
                B::Atom(99),
                B::And(atoms(20..22)),
            ]),
            B::Atom(3),
            B::Or(vec![B::Atom(7), B::Atom(8)]),
            B::Atom(4),
        ]),
        // Nested And-of-And (inner And is NOT an atom: general path) whose
        // result must still match appending its atoms in order.
        B::And(vec![B::Atom(0), B::And(atoms(1..5)), B::Atom(5)]),
        // Deep alternation.
        B::Or(vec![
            B::And(vec![
                B::Or(vec![B::And(atoms(0..3)), B::And(atoms(3..6))]),
                B::Atom(50),
                B::Or(vec![B::Atom(60), B::Atom(61), B::Atom(62)]),
            ]),
            B::Atom(70),
        ]),
        // Degenerate: single atom, empty And, empty Or.
        B::Atom(42),
        B::And(vec![]),
        B::Or(vec![]),
    ];

    let options = vnn_normalize::NormalizeOptions {
        max_clauses: 10_000,
        max_clause_len: 4096,
    };
    for (i, expr) in exprs.iter().enumerate() {
        let fast = vnn_normalize::to_dnf(expr, &options);
        let reference = dnf_reference::to_dnf(expr, &options);
        match (fast, reference) {
            (Ok(f), Ok(r)) => assert_eq!(f, r, "expr {i}: fast DNF diverged from reference"),
            (f, r) => panic!("expr {i}: outcome mismatch fast={f:?} reference={r:?}"),
        }
    }

    // Error parity: max_clause_len overflow through the ATOM fast path must
    // produce the reference's exact error string.
    let tight = vnn_normalize::NormalizeOptions {
        max_clauses: 8,
        max_clause_len: 3,
    };
    let overflow = B::And(atoms(0..5));
    let fast_err = vnn_normalize::to_dnf(&overflow, &tight).unwrap_err();
    let ref_err = dnf_reference::to_dnf(&overflow, &tight).unwrap_err();
    assert_eq!(fast_err.to_string(), ref_err.to_string());
}

/// A `0` / `0.0` output threshold must normalize to bit-exact **+0.0**.
///
/// `-constant / coeff` yields IEEE -0.0 for a zero literal, which is numerically
/// identical to +0.0 but not bit-identical. `beta_crown/verify/graph.rs` compares
/// these thresholds with `to_bits()`, so a -0.0 made the Cersyve conic-proof gate
/// unreachable from the parser on EVERY real property — it only ever matched a
/// hand-built fixture. This pins the canonicalization so the lane cannot silently
/// go dead again.
#[ntest::timeout(10000)]
#[test]
fn zero_literal_output_threshold_normalizes_to_positive_zero() {
    for literal in ["0", "0.0", "-0.0"] {
        let content = format!(
            "(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
             (assert (<= Y_0 {literal}))\n(assert (>= Y_1 {literal}))\n"
        );
        let spec = crate::vnnlib::parse_vnnlib(&content)
            .unwrap_or_else(|e| panic!("literal {literal}: {e}"));
        for constraint in &spec.output_constraints {
            let threshold = match constraint {
                OutputConstraint::LessEqConst(_, value)
                | OutputConstraint::GreaterEqConst(_, value) => value,
                other => panic!("literal {literal}: unexpected constraint {other:?}"),
            };
            assert_eq!(
                threshold.to_bits(),
                0.0f64.to_bits(),
                "literal {literal} produced a non-canonical zero ({threshold:?}, bits {:#x}); \
                 bit-exact consumers such as the Cersyve conic gate will not match it",
                threshold.to_bits()
            );
        }
    }
}

/// The zero canonicalization must not perturb any non-zero threshold.
#[ntest::timeout(10000)]
#[test]
fn nonzero_output_thresholds_survive_zero_sign_canonicalization() {
    let content = "(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
                   (assert (<= Y_0 -1.5))\n(assert (>= Y_1 2.25))\n";
    let spec = crate::vnnlib::parse_vnnlib(content).expect("parse");
    let mut seen = Vec::new();
    for constraint in &spec.output_constraints {
        match constraint {
            OutputConstraint::LessEqConst(_, value)
            | OutputConstraint::GreaterEqConst(_, value) => seen.push(value),
            other => panic!("unexpected constraint {other:?}"),
        }
    }
    seen.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert_eq!(
        seen,
        vec![&-1.5_f64, &2.25_f64],
        "non-zero thresholds must be untouched"
    );
}
