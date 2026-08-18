// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the star predicate LP.
//!
//! The load-bearing property is that the predicate is actually HONOURED: a star's box-α
//! interval bound ignores `A·α ≤ b`, so if the LP returns the same box it is not doing its
//! job and every downstream split-count claim is wrong.

use std::time::{Duration, Instant};

use super::{star_predicate_bounds, StarLpReport, StarLpRequest};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn solve(req: &StarLpRequest) -> StarLpReport {
    star_predicate_bounds(req, Duration::from_secs(5), deadline()).expect("star LP")
}

#[test]
fn unconstrained_star_reproduces_the_interval_bound() {
    // x = 1 + 2·a0 - 1·a1 over a in [-1,1]^2, no predicate.
    // Interval range is 1 ± 3 = [-2, 4]; with no constraints the LP must agree.
    let req = StarLpRequest {
        alpha_dim: 2,
        a_rows: vec![],
        b: vec![],
        targets: vec![(1.0, vec![2.0, -1.0])],
    };
    let rep = solve(&req);
    assert!(!rep.infeasible);
    let (lo, hi) = rep.lp_bounds[0];
    assert!((lo - -2.0).abs() < 1e-6, "lower {lo}");
    assert!((hi - 4.0).abs() < 1e-6, "upper {hi}");
}

#[test]
fn the_predicate_actually_tightens_the_bound() {
    // Same star, but constrain a0 <= -0.5. Now x = 1 + 2·a0 - a1 <= 1 - 1 + 1 = 1,
    // strictly tighter than the interval upper of 4. If this returns 4 the predicate
    // is being ignored and the whole point of the LP is lost.
    let req = StarLpRequest {
        alpha_dim: 2,
        a_rows: vec![vec![1.0, 0.0]],
        b: vec![-0.5],
        targets: vec![(1.0, vec![2.0, -1.0])],
    };
    let rep = solve(&req);
    assert!(!rep.infeasible);
    let (lo, hi) = rep.lp_bounds[0];
    assert!(
        hi <= 1.0 + 1e-6,
        "predicate ignored: upper {hi} should be <= 1"
    );
    assert!(lo >= -2.0 - 1e-6, "lower must stay sound: {lo}");
}

#[test]
fn contradictory_predicate_is_proved_infeasible() {
    // a0 <= -0.9 AND -a0 <= -0.9 (i.e. a0 >= 0.9) cannot both hold.
    let req = StarLpRequest {
        alpha_dim: 1,
        a_rows: vec![vec![1.0], vec![-1.0]],
        b: vec![-0.9, -0.9],
        targets: vec![(0.0, vec![1.0])],
    };
    let rep = solve(&req);
    assert!(
        rep.infeasible,
        "an empty predicate must be proved empty so the branch can be dropped"
    );
}

#[test]
fn multiple_targets_are_bounded_independently() {
    // x0 = a0, x1 = a1, constrained a0 + a1 <= -1.5.
    let req = StarLpRequest {
        alpha_dim: 2,
        a_rows: vec![vec![1.0, 1.0]],
        b: vec![-1.5],
        targets: vec![(0.0, vec![1.0, 0.0]), (0.0, vec![0.0, 1.0])],
    };
    let rep = solve(&req);
    assert!(!rep.infeasible);
    // a0 <= -1.5 - a1 <= -1.5 + 1 = -0.5 for both coordinates by symmetry.
    for (i, &(lo, hi)) in rep.lp_bounds.iter().enumerate() {
        assert!(hi <= -0.5 + 1e-6, "target {i} upper {hi} should be <= -0.5");
        assert!(
            lo >= -1.0 - 1e-6,
            "target {i} lower {lo} must respect the box"
        );
    }
}

#[test]
fn sound_bounds_never_narrow_an_independently_proven_box() {
    // A deliberately WRONG (too tight) LP box must not shrink the caller's own bound.
    let rep = StarLpReport {
        lp_bounds: vec![(0.4, 0.6)],
        infeasible: false,
    };
    let merged = rep.sound_bounds(&[(0.0, 1.0)]);
    assert_eq!(
        merged,
        vec![(0.0, 1.0)],
        "must keep the weaker (proven) side"
    );
}

#[test]
fn sound_bounds_falls_back_when_the_lp_is_non_finite() {
    let rep = StarLpReport {
        lp_bounds: vec![(f64::NEG_INFINITY, f64::INFINITY)],
        infeasible: false,
    };
    assert_eq!(rep.sound_bounds(&[(-2.0, 3.0)]), vec![(-2.0, 3.0)]);
}

#[test]
fn malformed_requests_fail_closed() {
    let bad_width = StarLpRequest {
        alpha_dim: 2,
        a_rows: vec![vec![1.0]],
        b: vec![0.0],
        targets: vec![],
    };
    assert!(star_predicate_bounds(&bad_width, Duration::from_secs(1), deadline()).is_err());

    let ragged_rhs = StarLpRequest {
        alpha_dim: 1,
        a_rows: vec![vec![1.0]],
        b: vec![],
        targets: vec![],
    };
    assert!(star_predicate_bounds(&ragged_rhs, Duration::from_secs(1), deadline()).is_err());

    let nan_target = StarLpRequest {
        alpha_dim: 1,
        a_rows: vec![],
        b: vec![],
        targets: vec![(f64::NAN, vec![1.0])],
    };
    assert!(star_predicate_bounds(&nan_target, Duration::from_secs(1), deadline()).is_err());
}

fn run_repeated_query_fixture(m: usize, k: usize, ntargets: usize, reps: u32) {
    use super::StarLpSession;

    let mut a_rows = Vec::new();
    let mut b = Vec::new();
    for i in 0..k {
        let mut row = vec![0.0; m];
        row[i % m] = if i % 2 == 0 { 1.0 } else { -1.0 };
        a_rows.push(row);
        b.push(0.9);
    }
    let targets: Vec<(f64, Vec<f64>)> = (0..ntargets)
        .map(|t| {
            let g = (0..m).map(|j| ((t + j) % 7) as f64 * 0.1 - 0.3).collect();
            (0.5, g)
        })
        .collect();
    let req = StarLpRequest {
        alpha_dim: m,
        a_rows,
        b,
        targets,
    };

    let t0 = Instant::now();
    let mut sess = StarLpSession::new(&req, Duration::from_secs(5), deadline()).expect("session");
    let build = t0.elapsed();

    let t1 = Instant::now();
    let got = sess.bounds(0).expect("query").expect("target 0 must exist");
    assert!(
        got.0.is_finite() && got.1.is_finite() && got.0 <= got.1,
        "cost probe returned invalid bounds {got:?}"
    );
    let first = t1.elapsed();

    let t2 = Instant::now();
    for _ in 0..reps {
        let repeated = sess
            .bounds(0)
            .expect("repeated query")
            .expect("target 0 must exist");
        assert!(
            (repeated.0 - got.0).abs() < 1e-9 && (repeated.1 - got.1).abs() < 1e-9,
            "repeated query changed its rigorous result: {repeated:?} vs {got:?}"
        );
    }
    let per = t2.elapsed() / reps;

    let t3 = Instant::now();
    let batch_report =
        star_predicate_bounds(&req, Duration::from_secs(5), deadline()).expect("batch");
    assert!(
        !batch_report.infeasible,
        "cost fixture must remain feasible"
    );
    let batch_bounds = batch_report.lp_bounds[0];
    assert!(
        got.0 <= batch_bounds.0 + 1e-6 && got.1 >= batch_bounds.1 - 1e-6,
        "session bounds {got:?} were tighter than batch bounds {batch_bounds:?}"
    );
    let batch = t3.elapsed();

    println!(
        "m={m} k={k} targets={ntargets}: build {build:?} | first {first:?} | repeat {per:?} | batch-wrapper {batch:?} | got {got:?}"
    );
}

/// Small deterministic repeated-query contract for the ordinary test gate.
#[test]
fn repeated_query_contract() {
    run_repeated_query_fixture(5, 4, 1, 2);
}

/// Expanded shape matrix. Timings are diagnostic only; the assertions remain
/// an ordinary, hermetic contract so the source-policy gate cannot skip it.
#[test]
fn lp_shape_matrix_contract() {
    for (m, k, ntargets) in [
        (5usize, 4usize, 1usize),
        (5, 4, 50),
        (5, 20, 50),
        (5, 40, 1),
    ] {
        run_repeated_query_fixture(m, k, ntargets, 20);
    }
}

#[test]
fn alpha_only_session_agrees_with_the_column_encoding() {
    use super::StarLpSession;
    let req = StarLpRequest {
        alpha_dim: 2,
        a_rows: vec![vec![1.0, 0.0]],
        b: vec![-0.5],
        targets: vec![(1.0, vec![2.0, -1.0])],
    };
    let batch = solve(&req);
    let mut sess =
        StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("session");
    let (lo, hi) = sess.expr_bounds(1.0, &[2.0, -1.0]).expect("expr bounds");
    println!(
        "column-encoding {:?} | alpha-only ({lo}, {hi})",
        batch.lp_bounds[0]
    );
    assert!(
        lo.is_finite() && hi.is_finite(),
        "alpha-only returned ({lo}, {hi})"
    );
    let (elo, ehi) = batch.lp_bounds[0];
    assert!(
        lo <= elo + 1e-6,
        "alpha-only lower {lo} tighter than exact {elo}"
    );
    assert!(
        hi >= ehi - 1e-6,
        "alpha-only upper {hi} tighter than exact {ehi}"
    );
    assert!(hi <= 1.0 + 1e-3, "predicate must still bite: {hi}");
}

/// Does the alpha-only session agree with the column encoding as a star ACCUMULATES
/// predicate rows, which is what happens along a driver branch?
#[test]
fn alpha_only_agrees_as_predicate_rows_accumulate() {
    use super::StarLpSession;
    let mut a_rows: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let target = (0.25f64, vec![1.0f64, -0.5, 0.75]);

    for step in 0..6 {
        // Append a row, the way an exact ReLU split does.
        let mut row = vec![0.0; 3];
        row[step % 3] = if step % 2 == 0 { 1.0 } else { -1.0 };
        a_rows.push(row);
        b.push(0.6);

        let req = StarLpRequest {
            alpha_dim: 3,
            a_rows: a_rows.clone(),
            b: b.clone(),
            targets: vec![target.clone()],
        };
        let col = solve(&req).lp_bounds[0];
        let mut sess = StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline())
            .expect("session");
        let alpha = sess.expr_bounds(target.0, &target.1).expect("expr");
        println!("k={}: column {col:?} | alpha-only {alpha:?}", a_rows.len());
        assert!(
            alpha.0.is_finite() && alpha.1.is_finite(),
            "k={}: alpha-only went non-finite: {alpha:?}",
            a_rows.len()
        );
        assert!(
            alpha.0 <= col.0 + 1e-6 && alpha.1 >= col.1 - 1e-6,
            "k={}: tighter than exact",
            a_rows.len()
        );
    }
}

/// Reusing ONE alpha-only session for MANY different objectives — the driver's pattern.
#[test]
fn alpha_only_session_is_reusable_across_objectives() {
    use super::StarLpSession;
    let req = StarLpRequest {
        alpha_dim: 3,
        a_rows: vec![vec![1.0, 1.0, 0.0], vec![0.0, -1.0, 1.0]],
        b: vec![0.4, 0.3],
        targets: vec![],
    };
    let mut sess =
        StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("session");

    for t in 0..8 {
        let g: Vec<f64> = (0..3).map(|j| ((t + j) % 5) as f64 * 0.3 - 0.6).collect();
        let c = 0.1 * t as f64;
        let reused = sess.expr_bounds(c, &g).expect("reused");

        // Fresh session, same question.
        let mut fresh =
            StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("fresh");
        let once = fresh.expr_bounds(c, &g).expect("fresh query");
        println!("t={t}: reused {reused:?} | fresh {once:?}");
        assert!(
            (reused.0 - once.0).abs() < 1e-9 && (reused.1 - once.1).abs() < 1e-9,
            "t={t}: session reuse changed the answer: {reused:?} vs {once:?}"
        );
    }
}

/// Repeated alpha-only/column query contract; timings remain diagnostic only.
#[test]
fn expr_vs_column_cost() {
    use super::StarLpSession;
    for (m, k) in [(5usize, 4usize), (5, 20), (5, 40), (10, 40)] {
        let mut a_rows = Vec::new();
        let mut b = Vec::new();
        for i in 0..k {
            let mut row = vec![0.0; m];
            row[i % m] = if i % 2 == 0 { 1.0 } else { -1.0 };
            a_rows.push(row);
            b.push(0.9);
        }
        let g: Vec<f64> = (0..m).map(|j| (j % 7) as f64 * 0.1 - 0.3).collect();

        // alpha-only: one session, many objectives
        let req_a = StarLpRequest {
            alpha_dim: m,
            a_rows: a_rows.clone(),
            b: b.clone(),
            targets: vec![],
        };
        let mut sa =
            StarLpSession::new_alpha_only(&req_a, Duration::from_secs(5), deadline()).expect("a");
        let t = Instant::now();
        let reps = 20;
        let mut expr = None;
        for _ in 0..reps {
            expr = Some(sa.expr_bounds(0.5, &g).expect("expression query"));
        }
        let expr_per = t.elapsed() / reps;

        // column: fresh 1-target session each time (what the driver does)
        let req_c = StarLpRequest {
            alpha_dim: m,
            a_rows,
            b,
            targets: vec![(0.5, g.clone())],
        };
        let t = Instant::now();
        let mut column = None;
        for _ in 0..reps {
            let mut sc = StarLpSession::new(&req_c, Duration::from_secs(5), deadline()).expect("c");
            column = Some(
                sc.bounds(0)
                    .expect("column query")
                    .expect("target 0 must exist"),
            );
        }
        let col_per = t.elapsed() / reps;

        let expr = expr.expect("expression loop ran");
        let column = column.expect("column loop ran");
        assert!(expr.0.is_finite() && expr.1.is_finite() && expr.0 <= expr.1);
        assert!(column.0.is_finite() && column.1.is_finite() && column.0 <= column.1);
        assert!(
            expr.0 <= column.0 + 1e-6 && expr.1 >= column.1 - 1e-6,
            "alpha-only expression {expr:?} was tighter than column encoding {column:?}"
        );

        println!("m={m} k={k}: expr {expr_per:?} | column {col_per:?}");
    }
}

/// Driver-realistic predicate: float coefficients of mixed magnitude, as produced by an
/// exact ReLU split (row = the generator row, rhs = -center).
#[test]
fn alpha_only_handles_driver_realistic_coefficients() {
    use super::StarLpSession;
    // These are the exact bit patterns an exact ReLU split produces; `FRAC_1_SQRT_2` differs
    // in the last ulp, so the literal stays as-is.
    #[allow(clippy::approx_constant)]
    let a_rows = vec![
        vec![
            0.31622776601683794,
            -0.7071067811865476,
            0.1414213562373095,
            0.0,
            0.9486832980505138,
        ],
        vec![
            -0.05773502691896258,
            0.2886751345948129,
            -0.8660254037844386,
            0.11547005383792515,
            0.0,
        ],
        vec![
            1.2247448713915892e-3,
            0.0,
            0.0,
            -4.898979485566356e-2,
            0.24494897427831783,
        ],
    ];
    let b = vec![
        -0.12309149097933272,
        0.4472135954999579,
        -1.8973665961010275e-3,
    ];
    let g = vec![
        0.7745966692414834,
        -0.2581988897471611,
        0.5163977794943222,
        0.0,
        -0.12909944487358055,
    ];

    let req = StarLpRequest {
        alpha_dim: 5,
        a_rows,
        b,
        targets: vec![(0.31, g.clone())],
    };
    let col = solve(&req).lp_bounds[0];
    let mut sess =
        StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("session");
    let alpha = sess.expr_bounds(0.31, &g).expect("expr");
    println!("realistic: column {col:?} | alpha-only {alpha:?}");
    assert!(
        alpha.0.is_finite() && alpha.1.is_finite(),
        "alpha-only DECLINED on driver-realistic data: {alpha:?} (column gave {col:?})"
    );
}

/// A row that is entirely zero — can arise if a split lands on a coordinate with no
/// generator dependence. An empty coefficient list may be a degenerate row for AY.
#[test]
fn alpha_only_handles_a_zero_row() {
    use super::StarLpSession;
    let req = StarLpRequest {
        alpha_dim: 3,
        a_rows: vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
        b: vec![0.5, 0.4],
        targets: vec![],
    };
    let mut sess =
        StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("session");
    let got = sess.expr_bounds(0.0, &[1.0, 1.0, 1.0]).expect("expr");
    println!("zero-row: {got:?}");
    assert!(
        got.0.is_finite() && got.1.is_finite(),
        "declined on a zero row: {got:?}"
    );
}

/// The untrusted-solver / trusted-verifier path must never be TIGHTER than the exact LP —
/// that is the whole soundness claim. It is allowed to be looser.
#[test]
fn verified_float_bounds_are_sound_against_the_exact_lp() {
    use super::StarLpSession;
    // Same split-derived fixtures as above: keep the literals, not `FRAC_1_SQRT_2`.
    #[allow(clippy::approx_constant)]
    let cases: Vec<(f64, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> = vec![
        (1.0, vec![2.0, -1.0], vec![vec![1.0, 0.0]], vec![-0.5]),
        (0.0, vec![1.0, 1.0], vec![vec![1.0, 1.0]], vec![-1.5]),
        (
            0.31,
            vec![
                0.7745966692414834,
                -0.2581988897471611,
                0.5163977794943222,
                0.0,
                -0.12909944487358055,
            ],
            vec![
                vec![
                    0.31622776601683794,
                    -0.7071067811865476,
                    0.1414213562373095,
                    0.0,
                    0.9486832980505138,
                ],
                vec![
                    -0.05773502691896258,
                    0.2886751345948129,
                    -0.8660254037844386,
                    0.11547005383792515,
                    0.0,
                ],
            ],
            vec![-0.12309149097933272, 0.4472135954999579],
        ),
    ];
    let mut published = 0usize;
    for (c, g, a, b) in cases {
        let req = StarLpRequest {
            alpha_dim: g.len(),
            a_rows: a.clone(),
            b: b.clone(),
            targets: vec![(c, g.clone())],
        };
        let (elo, ehi) = solve(&req).lp_bounds[0];
        let mut sess = StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline())
            .expect("session");
        match sess.verified_float_bounds(c, &g, &a, &b).expect("no error") {
            Some((lo, hi)) => {
                published += 1;
                println!("exact ({elo}, {ehi}) | verified-float ({lo}, {hi})");
                assert!(
                    lo <= elo + 1e-6,
                    "verified lower {lo} TIGHTER than exact {elo}"
                );
                assert!(
                    hi >= ehi - 1e-6,
                    "verified upper {hi} TIGHTER than exact {ehi}"
                );
            }
            None => println!("exact ({elo}, {ehi}) | verified-float declined"),
        }
    }
    assert!(
        published > 0,
        "verified-float declined every fixture; the soundness assertions were vacuous"
    );
}

#[test]
fn verified_float_declines_a_constant_objective_explicitly() {
    use super::StarLpSession;
    let req = StarLpRequest {
        alpha_dim: 2,
        a_rows: vec![],
        b: vec![],
        targets: vec![],
    };
    let mut sess =
        StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("session");
    assert_eq!(
        sess.verified_float_bounds(1.25, &[0.0, 0.0], &[], &[])
            .expect("constant query"),
        None,
        "the optional fast lane must explicitly decline an empty objective"
    );
    assert_eq!(
        sess.expr_bounds(1.25, &[0.0, 0.0])
            .expect("rigorous constant fallback"),
        (1.25, 1.25),
        "the rigorous fallback must cover the declined constant objective"
    );
}

/// Repeated verified-float/rigorous query contract; timings remain diagnostic.
#[test]
fn verified_float_cost() {
    use super::StarLpSession;
    let mut published = 0usize;
    for (m, k) in [(5usize, 4usize), (5, 20), (5, 40)] {
        let mut a_rows = Vec::new();
        let mut b = Vec::new();
        for i in 0..k {
            let mut row = vec![0.0; m];
            row[i % m] = if i % 2 == 0 { 1.0 } else { -1.0 };
            // Tight, near-degenerate rows — the regime that forces the exact rim.
            a_rows.push(row);
            b.push(-0.85 + 0.01 * i as f64);
        }
        let g: Vec<f64> = (0..m).map(|j| (j % 7) as f64 * 0.1 - 0.3).collect();
        let req = StarLpRequest {
            alpha_dim: m,
            a_rows: a_rows.clone(),
            b: b.clone(),
            targets: vec![(0.5, g.clone())],
        };
        let mut sess =
            StarLpSession::new_alpha_only(&req, Duration::from_secs(5), deadline()).expect("s");

        let reps = 20;
        let t = Instant::now();
        let mut got = None;
        for _ in 0..reps {
            got = sess.verified_float_bounds(0.5, &g, &a_rows, &b).expect("q");
        }
        let vf = t.elapsed() / reps;

        let t = Instant::now();
        let mut rigorous = None;
        for _ in 0..reps {
            rigorous = Some(sess.expr_bounds(0.5, &g).expect("rigorous query"));
        }
        let rig = t.elapsed() / reps;

        if let Some((lo, hi)) = got {
            let (exact_lo, exact_hi) = rigorous.expect("rigorous loop ran");
            assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
            assert!(
                lo <= exact_lo + 1e-6 && hi >= exact_hi - 1e-6,
                "verified-float ({lo}, {hi}) was tighter than rigorous ({exact_lo}, {exact_hi})"
            );
            published += 1;
        }

        println!("m={m} k={k}: verified-float {vf:?} {got:?} | rigorous-expr {rig:?}");
    }
    assert!(
        published > 0,
        "verified-float cost probe declined every configuration"
    );
}
