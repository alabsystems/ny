// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The load-bearing test here is the CROSS-CHECK against the exact LP: this bound is allowed
//! to be looser than the truth, never tighter. A tighter-than-true bound would skip a needed
//! ReLU split and could turn a real counterexample into a false "verified".

use std::time::{Duration, Instant};

use super::{dual_certifies_empty, dual_coordinate_bounds};
use crate::star_lp::{star_predicate_bounds, StarLpRequest};

fn exact(c: f64, g: &[f64], a: &[Vec<f64>], b: &[f64]) -> (f64, f64) {
    let req = StarLpRequest {
        alpha_dim: g.len(),
        a_rows: a.to_vec(),
        b: b.to_vec(),
        targets: vec![(c, g.to_vec())],
    };
    let rep = star_predicate_bounds(
        &req,
        Duration::from_secs(5),
        Instant::now() + Duration::from_secs(30),
    )
    .expect("exact LP");
    rep.lp_bounds[0]
}

#[test]
fn with_no_constraints_it_is_exactly_the_interval_bound() {
    // lambda = 0 is optimal when there is nothing to dualise.
    let (lo, hi) = dual_coordinate_bounds(1.0, &[2.0, -1.0], &[], &[], 0);
    assert!((lo - -2.0).abs() < 1e-6, "lower {lo}");
    assert!((hi - 4.0).abs() < 1e-6, "upper {hi}");
}

#[test]
fn it_never_reports_tighter_than_the_exact_lp() {
    // A spread of predicates; the dual must bracket the exact answer from OUTSIDE.
    let cases: Vec<(f64, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> = vec![
        (1.0, vec![2.0, -1.0], vec![vec![1.0, 0.0]], vec![-0.5]),
        (0.0, vec![1.0, 1.0], vec![vec![1.0, 1.0]], vec![-1.5]),
        (
            -2.0,
            vec![0.5, 2.0, -1.0],
            vec![vec![1.0, -1.0, 0.0]],
            vec![0.25],
        ),
        (
            3.0,
            vec![1.0, -2.0, 0.5],
            vec![vec![1.0, 1.0, 1.0], vec![-1.0, 0.0, 1.0]],
            vec![0.5, -0.25],
        ),
    ];
    for (c, g, a, b) in cases {
        let (elo, ehi) = exact(c, &g, &a, &b);
        let (dlo, dhi) = dual_coordinate_bounds(c, &g, &a, &b, 60);
        assert!(
            dlo <= elo + 1e-6,
            "dual lower {dlo} is TIGHTER than exact {elo} — unsound"
        );
        assert!(
            dhi >= ehi - 1e-6,
            "dual upper {dhi} is TIGHTER than exact {ehi} — unsound"
        );
    }
}

#[test]
fn ascent_actually_tightens_relative_to_the_box() {
    // Same case as the star_lp test: a0 <= -0.5 pulls the upper from 4 down toward 1.
    let (c, g) = (1.0, vec![2.0, -1.0]);
    let a = vec![vec![1.0, 0.0]];
    let b = vec![-0.5];
    let (_, hi0) = dual_coordinate_bounds(c, &g, &a, &b, 0);
    let (_, hi_n) = dual_coordinate_bounds(c, &g, &a, &b, 200);
    assert!(
        (hi0 - 4.0).abs() < 1e-6,
        "zero-step must equal the box: {hi0}"
    );
    assert!(
        hi_n < hi0 - 0.5,
        "ascent must exploit the predicate: {hi_n} vs box {hi0}"
    );
    assert!(
        hi_n >= 1.0 - 1e-6,
        "but never below the truth (1.0): {hi_n}"
    );
}

#[test]
fn zero_iterations_is_still_sound() {
    // Soundness must not depend on the ascent running at all.
    let a = vec![vec![1.0, 0.0]];
    let b = vec![-0.5];
    let (lo, hi) = dual_coordinate_bounds(1.0, &[2.0, -1.0], &a, &b, 0);
    let (elo, ehi) = exact(1.0, &[2.0, -1.0], &a, &b);
    assert!(lo <= elo + 1e-6 && hi >= ehi - 1e-6);
}

#[test]
fn a_contradictory_predicate_is_certified_empty() {
    // a0 <= -0.9 and -a0 <= -0.9 cannot both hold.
    let a = vec![vec![1.0], vec![-1.0]];
    let b = vec![-0.9, -0.9];
    assert!(dual_certifies_empty(&a, &b, 200));
}

#[test]
fn a_feasible_predicate_is_never_certified_empty() {
    // Dropping a reachable branch is the one unrecoverable error here, so this must hold
    // for every iteration count.
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![0.5, 0.5];
    for iters in [0, 1, 10, 200, 2000] {
        assert!(
            !dual_certifies_empty(&a, &b, iters),
            "falsely certified a FEASIBLE predicate empty at {iters} iters"
        );
    }
}

#[test]
fn an_empty_or_malformed_predicate_is_not_certified() {
    assert!(
        !dual_certifies_empty(&[], &[], 100),
        "no rows proves nothing"
    );
    let ragged = vec![vec![1.0, 0.0], vec![1.0]];
    assert!(!dual_certifies_empty(&ragged, &[0.0, 0.0], 100));
}

#[test]
fn malformed_inputs_fall_back_to_the_trivial_interval() {
    let (lo, hi) = dual_coordinate_bounds(0.0, &[1.0, 2.0], &[vec![1.0]], &[0.0], 10);
    assert_eq!((lo, hi), (f64::NEG_INFINITY, f64::INFINITY));
    let (lo, hi) = dual_coordinate_bounds(f64::NAN, &[1.0], &[], &[], 10);
    assert_eq!((lo, hi), (f64::NEG_INFINITY, f64::INFINITY));
}
