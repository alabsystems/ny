// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;
use std::time::{Duration, Instant};

use super::*;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn read(text: &str) -> Wcnf {
    Wcnf::read(Cursor::new(text), deadline()).unwrap()
}

#[test]
fn legacy_and_modern_preserve_the_same_original_clause_semantics() {
    let modern = read("h 1 -2 0\n3 -1 0\n5 2 0\n");
    let legacy = read("p wcnf 2 3 10\n10 1 -2 0\n3 -1 0\n5 2 0\n");
    assert_eq!(modern, legacy);
    assert_eq!(modern.validate_values(&[1.0, 0.0]).unwrap().cost, 8);
    assert_eq!(modern.validate_values(&[0.0, 0.0]).unwrap().cost, 5);
    assert!(modern.validate_values(&[0.0, 1.0]).is_err());
}

#[test]
fn whitespace_multiline_clauses_and_unused_variables_are_retained() {
    let formula = read(" c comment\r\np wcnf 3 2\n2 1\nc continued\n-2 0 4 3 0\n");
    assert_eq!(formula.variables(), 3);
    assert_eq!(formula.clauses[0].literals, [1, -2]);
    assert_eq!(formula.validate_values(&[0.0, 1.0, 0.0]).unwrap().cost, 6);
    assert!(formula.validate_values(&[0.0, 1.0]).is_err());
}

#[test]
fn integer_boundary_is_exact_and_never_rounded() {
    assert_eq!(read("9007199254740992 1 0\n").soft_total, MAX_EXACT_INTEGER);
    for text in [
        "9007199254740993 1 0\n",
        "9007199254740992 1 0\n1 -1 0\n",
        "p wcnf 1 1 9007199254740993\n1 1 0\n",
        "18446744073709551616 1 0\n",
    ] {
        assert!(Wcnf::read(Cursor::new(text), deadline()).is_err(), "{text}");
    }
}

#[test]
fn malformed_unsupported_and_truncated_inputs_refuse() {
    for text in [
        "",
        "c only a comment\n",
        "p cnf 1 1\n1 0\n",
        "p wcnf 1 2\n1 1 0\n",
        "p wcnf 1 1\n1 2 0\n",
        "p wcnf 1 1 0\n1 1 0\n",
        "0 1 0\n",
        "-1 1 0\n",
        "1 1\n",
        "1 1 0 garbage\n",
        "h -2147483648 0\n",
        "1 1 0\np wcnf 1 1\n",
        "p wcnf 2000001 0\n",
        "p wcnf 1 4000001\n",
    ] {
        assert!(Wcnf::read(Cursor::new(text), deadline()).is_err(), "{text}");
    }
}

#[test]
fn duplicates_tautologies_and_empty_clauses_do_not_disappear() {
    let formula = read("h 1 -1 0\n7 0\n3 1 1 0\n");
    assert_eq!(formula.clauses.len(), 3);
    assert_eq!(formula.validate_values(&[0.0]).unwrap().cost, 10);
    assert_eq!(formula.validate_values(&[1.0]).unwrap().cost, 7);
    assert!(read("p wcnf 0 1\nh 0\n").validate_values(&[]).is_err());
    assert_eq!(read("p wcnf 0 0\n").validate_values(&[]).unwrap().cost, 0);
}

#[test]
fn non_boolean_nonfinite_or_incomplete_candidates_refuse() {
    let formula = read("h 1 0\n3 -1 0\n");
    for values in [
        vec![],
        vec![0.5],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1.0, 0.0],
        vec![0.0],
    ] {
        assert!(formula.validate_values(&values).is_err());
    }
    assert_eq!(formula.validate_values(&[1.0]).unwrap().cost, 3);
}

fn encoded_costs(formula: &Wcnf, assignment: &[f64]) -> Vec<u64> {
    let parts = formula.encode(deadline()).unwrap();
    let soft: Vec<_> = formula.clauses.iter().filter_map(|c| c.weight).collect();
    let objective = parts.output_vars[0].0;
    let mut costs = Vec::new();
    for mask in 0..(1 << soft.len()) {
        let mut values = vec![0.0; parts.num_cols];
        values[..assignment.len()].copy_from_slice(assignment);
        let mut cost = 0;
        for (index, &weight) in soft.iter().enumerate() {
            if mask & (1 << index) != 0 {
                values[assignment.len() + index] = 1.0;
                cost += weight;
            }
        }
        values[objective] = cost as f64;
        let feasible = parts.problem.rows().iter().all(|r| {
            let value: f64 = r.coeffs.iter().map(|&(col, a)| a * values[col]).sum();
            r.lb <= value && value <= r.ub
        });
        if feasible {
            costs.push(cost);
        }
    }
    costs
}

#[test]
fn exhaustive_tiny_milp_assignments_match_original_clause_costs() {
    for text in [
        "h 1 -2 0\n3 -1 0\n5 2 0\n",
        "2 1 -1 0\n3 0\n4 -1 -1 0\n",
        "h 0\n2 1 0\n",
        "p wcnf 2 0\n",
    ] {
        let formula = read(text);
        for mask in 0..(1 << formula.variables()) {
            let values: Vec<_> = (0..formula.variables())
                .map(|i| if mask & (1 << i) == 0 { 0.0 } else { 1.0 })
                .collect();
            let actual = encoded_costs(&formula, &values).into_iter().min();
            let expected = formula.validate_values(&values).ok().map(|c| c.cost);
            assert_eq!(actual, expected, "{text}: {values:?}");
        }
    }
}

#[test]
fn backend_objective_and_dual_do_not_override_original_cost() {
    let formula = read("h 1 0\n7 -1 0\n");
    let result = MipResult::Sat {
        objective: -123.0,
        output_values: vec![-123.0],
        input_values: vec![1.0],
        dual_bound: Some(-123.0),
    };
    let Outcome::Incumbent(candidate) = formula.interpret_result(result, deadline()).unwrap()
    else {
        panic!("missing candidate")
    };
    assert_eq!(candidate.cost, 7);
}

#[test]
fn backend_nonincumbents_never_become_unsat_or_optimum_claims() {
    let formula = read("1 1 0\n");
    for result in [
        MipResult::Unsat { certified: true },
        MipResult::Unsat { certified: false },
        MipResult::Timeout,
        MipResult::Error("test refusal".into()),
    ] {
        assert!(matches!(
            formula.interpret_result(result, deadline()).unwrap(),
            Outcome::Unknown(_)
        ));
    }
}

#[test]
fn expired_deadline_prevents_parsing_encoding_and_publication() {
    let expired = Instant::now();
    assert!(matches!(
        Wcnf::read(Cursor::new("1 1 0\n"), expired),
        Err(Error::Deadline)
    ));
    let formula = read("1 1 0\n");
    assert!(matches!(formula.encode(expired), Err(Error::Deadline)));
    assert!(matches!(
        formula.interpret_result(MipResult::Timeout, expired),
        Err(Error::Deadline)
    ));
}
