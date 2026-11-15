// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn selected_binary_relaxation_preserves_all_bounds_rows_and_margin() {
    let mut p = MilpProblem::new();
    let a = p.add_integer_col(2.0, 0.0, 1.0);
    let b = p.add_integer_col(-3.0, 0.0, 1.0);
    p.add_margin_row(f64::NEG_INFINITY, 0.25, [(a, 1.0), (b, -1.0)])
        .unwrap();
    let before = p.clone();
    assert!(p.relax_binary_integrality(&[a]));
    assert_eq!(p.rows(), before.rows());
    assert_eq!(p.margin_row(), before.margin_row());
    let mut expected = before.cols().to_vec();
    expected[a.0].integer = false;
    assert_eq!(p.cols(), expected);
    assert!(p.cols()[b.0].integer);
}

#[test]
fn invalid_binary_relaxation_is_transactional() {
    let mut p = MilpProblem::new();
    let a = p.add_integer_col(0.0, 0.0, 1.0);
    let continuous = p.add_col(0.0, 0.0, 1.0);
    let nonbinary = p.add_integer_col(0.0, 0.0, 2.0);
    for cols in [
        vec![a, Col(999)],
        vec![a, continuous],
        vec![a, nonbinary],
        vec![a, a],
    ] {
        let before = p.clone();
        assert!(!p.relax_binary_integrality(&cols));
        assert_eq!(p, before);
    }
}

#[test]
fn pinned_or_already_relaxed_columns_refuse_without_changing_others() {
    let mut p = MilpProblem::new();
    let a = p.add_integer_col(0.0, 0.0, 1.0);
    let b = p.add_integer_col(0.0, 0.0, 1.0);
    p.fix_col(b, 1.0);
    let before = p.clone();
    assert!(!p.relax_binary_integrality(&[a, b]));
    assert_eq!(p, before);
    assert!(p.relax_binary_integrality(&[a]));
    let relaxed = p.clone();
    assert!(!p.relax_binary_integrality(&[a]));
    assert_eq!(p, relaxed);
}
