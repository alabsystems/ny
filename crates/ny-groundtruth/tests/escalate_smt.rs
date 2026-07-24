// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Route B integration tests: SMT escalation against a live `ay` solver
//! (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`).
//!
//! Every test degrades to a skip (with a stderr note) when no `ay` binary is
//! reachable (`$NY_AY`, `PATH`, or the Trust stage2 sysroot), so the suite
//! stays green on machines without the solver.
//!
//! The centrepiece is [`crown_unknown_smt_proves_quadratic_dominance`]: a
//! case where the CROWN relaxation of the quadratic ground-truth side is too
//! loose to decide (`PowConstant` secant looseness at the *interior* binding
//! point), but the exact QF_NRA query refutes the violation outright.

use ndarray::{Array1, Array2};
use ny_core::Bound;
use ny_groundtruth::{
    signed_plane_distance, sphere_residual, verify_against_ground_truth, EscalateError,
    EscalateOptions, GroundTruthOutcome, Relation, SmtEscalation, SmtVerdict,
};
use ny_propagate::layers::{LinearLayer, PowConstantLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

/// Locate `ay` or skip the test honestly.
fn solver() -> Option<SmtEscalation> {
    let located = SmtEscalation::locate();
    if located.is_none() {
        eprintln!(
            "skipping SMT escalation test: no `ay` solver (set NY_AY, PATH, or rustup trust)"
        );
    }
    located
}

/// f(x) = w·(|x0| + |x1| + |x2|) + bias as a genuine 3 -> 6 -> 1 FC-ReLU
/// network (each |xi| = relu(xi) + relu(−xi)).
fn abs_net(weight: f32, bias: f32) -> GraphNetwork {
    let w1 = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 0.0, 0.0, //
            -1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, -1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, -1.0,
        ],
    )
    .expect("shape");
    let w2 = Array2::from_shape_vec((1, 6), vec![weight; 6]).expect("shape");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, None).expect("valid linear")),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["lin1".to_string()],
    ));
    g.add_node(GraphNode::new(
        "readout",
        Layer::Linear(LinearLayer::new(w2, Some(Array1::from(vec![bias]))).expect("valid linear")),
        vec!["relu".to_string()],
    ));
    g.set_output("readout");
    g
}

fn unit_box() -> Vec<Bound> {
    vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]
}

#[test]
fn plane_case_crown_and_smt_agree() {
    // M0-style case: f = |x0|+|x1|+|x2| + 3 dominates the plane g = x2 − 1/2
    // with margin >= 5/2. CROWN proves it; the exact QF_LRA query must agree
    // and return an Alethe certificate.
    let f = abs_net(1.0, 3.0);
    let g = signed_plane_distance([0.0, 0.0, 1.0], -0.5).expect("plane builds");
    let bounds = unit_box();

    let crown =
        verify_against_ground_truth(&f, &g, Relation::Dominates, &bounds).expect("crown path runs");
    assert!(
        matches!(crown, GroundTruthOutcome::Verified { .. }),
        "CROWN must verify the plane case, got {crown:?}"
    );

    let Some(solver) = solver() else { return };
    let verdict = solver
        .escalate(
            &f,
            &g,
            Relation::Dominates,
            &bounds,
            &EscalateOptions::default(),
        )
        .expect("escalation runs");
    match verdict {
        SmtVerdict::Proved { certificate, query } => {
            assert!(query.is_file(), "query file kept for audit");
            let cert = certificate.expect("ay emits Alethe proofs by default");
            assert!(cert.is_file(), "certificate file exists");
        }
        other => panic!("SMT must agree with CROWN (Proved), got {other:?}"),
    }
}

#[test]
fn crown_unknown_smt_proves_quadratic_dominance() {
    // f = 2(|x0|+|x1|+|x2|) − 3/4 vs g = ‖x‖² − 1 on [−1,1]² × {0}:
    // h(x) = Σ (2|xi| − xi²) + 1/4, minimum 1/4 > 0 at the box *centre*.
    // CROWN cannot prove it: the lower relaxation of the −xi² terms is the
    // secant (exact only at the endpoints, off by 1 at the interior binding
    // point) and the |xi| lower envelope is linear, so the certified lower
    // bound is ≤ −3/4 + eps. The exact NRA query refutes h < 0 outright.
    let f = abs_net(2.0, -0.75);
    let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere builds");
    let bounds = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(0.0, 0.0),
    ];

    let crown =
        verify_against_ground_truth(&f, &g, Relation::Dominates, &bounds).expect("crown path runs");
    assert!(
        matches!(crown, GroundTruthOutcome::Unknown { .. }),
        "CROWN must be too loose here (else the test lost its point), got {crown:?}"
    );

    let Some(solver) = solver() else { return };
    let verdict = solver
        .escalate(
            &f,
            &g,
            Relation::Dominates,
            &bounds,
            &EscalateOptions::default(),
        )
        .expect("escalation runs");
    match verdict {
        SmtVerdict::Proved { certificate, query } => {
            assert!(query.is_file());
            let cert = certificate.expect("ay emits Alethe proofs by default");
            assert!(cert.is_file(), "Alethe certificate exists on disk");
        }
        other => panic!("SMT must decide what CROWN could not, got {other:?}"),
    }
}

#[test]
fn falsified_quadratic_returns_validated_witness() {
    // f = 2(|x0|+|x1|+|x2|) − 5/4 vs the same sphere: h(0) = −1/4 < 0, so
    // dominance is false. The sat model must validate in exact rational
    // arithmetic before being reported.
    let Some(solver) = solver() else { return };
    let f = abs_net(2.0, -1.25);
    let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere builds");
    let bounds = unit_box();

    let verdict = solver
        .escalate(
            &f,
            &g,
            Relation::Dominates,
            &bounds,
            &EscalateOptions::default(),
        )
        .expect("escalation runs");
    match verdict {
        SmtVerdict::Falsified {
            witness,
            witness_exact,
            output_index,
            difference_exact,
            ..
        } => {
            assert_eq!(witness.len(), 3);
            assert_eq!(witness_exact.len(), 3);
            assert_eq!(output_index, 0);
            assert!(
                witness
                    .iter()
                    .all(|w| w.is_finite() && (-1.0..=1.0).contains(w)),
                "witness stays in the box: {witness:?}"
            );
            assert!(
                difference_exact.starts_with('-'),
                "violating h value is negative: {difference_exact}"
            );
        }
        // A placeholder model that fails exact validation would come back as
        // ViolationExists; for this query the violation region has interior
        // (h < 0 on a neighbourhood of 0), so ay finds a rational witness.
        other => panic!("expected a validated witness, got {other:?}"),
    }
}

#[test]
fn absbound_escalation_proves_self_equivalence() {
    // |f − f| = 0 <= 1/2 everywhere: the AbsBound violation query is unsat.
    let Some(solver) = solver() else { return };
    let f = abs_net(1.0, 0.0);
    let verdict = solver
        .escalate(
            &f,
            &f,
            Relation::AbsBound(0.5),
            &unit_box(),
            &EscalateOptions::default(),
        )
        .expect("escalation runs");
    assert!(
        matches!(verdict, SmtVerdict::Proved { .. }),
        "self-equivalence within eps must be Proved, got {verdict:?}"
    );
}

#[test]
fn strict_margin_flag_flips_a_touching_case() {
    // f = |x0|+|x1|+|x2| dominates g = plane(x2) − 0 ... f − g = Σ|xi| − x2
    // touches 0 exactly at x = 0. Non-strict dominance holds (unsat);
    // requiring a strict margin makes the violation h <= 0 satisfiable at 0,
    // which validates exactly.
    let Some(solver) = solver() else { return };
    let f = abs_net(1.0, 0.0);
    let g = signed_plane_distance([0.0, 0.0, 1.0], 0.0).expect("plane builds");
    let bounds = unit_box();

    let non_strict = solver
        .escalate(
            &f,
            &g,
            Relation::Dominates,
            &bounds,
            &EscalateOptions::default(),
        )
        .expect("escalation runs");
    assert!(
        matches!(non_strict, SmtVerdict::Proved { .. }),
        "f >= g holds (touching at 0), got {non_strict:?}"
    );

    let strict = solver
        .escalate(
            &f,
            &g,
            Relation::Dominates,
            &bounds,
            &EscalateOptions {
                require_strict_margin: true,
                ..EscalateOptions::default()
            },
        )
        .expect("escalation runs");
    assert!(
        matches!(
            strict,
            SmtVerdict::Falsified { .. } | SmtVerdict::ViolationExists { .. }
        ),
        "strict margin fails at the touching point, got {strict:?}"
    );
}

#[test]
fn unsupported_shapes_error_before_reaching_the_solver() {
    // No solver needed: these must fail at encoding time, so use a dummy
    // binary path — reaching the subprocess would be the bug.
    let solver = SmtEscalation::with_solver(
        std::path::PathBuf::from("/nonexistent/ay"),
        std::env::temp_dir(),
    );

    // Fractional PowConstant exponent.
    let mut f = GraphNetwork::new();
    f.add_node(GraphNode::from_input(
        "half_pow",
        Layer::PowConstant(PowConstantLayer::new(0.5)),
    ));
    f.set_output("half_pow");
    let g = signed_plane_distance([1.0, 0.0, 0.0], 0.0).expect("plane builds");
    // f outputs 3 values, g outputs 1 — use f on both sides to keep shapes
    // legal and hit the exponent gate.
    let err = solver
        .escalate(
            &f,
            &f,
            Relation::Dominates,
            &unit_box(),
            &EscalateOptions::default(),
        )
        .expect_err("fractional exponent must be refused");
    assert!(
        matches!(err, EscalateError::UnsupportedExponent { .. }),
        "got {err:?}"
    );

    // Non-finite input box.
    let err = solver
        .escalate(
            &g,
            &g,
            Relation::Dominates,
            &[
                Bound::new(-1.0, 1.0),
                Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
                Bound::new(-1.0, 1.0),
            ],
            &EscalateOptions::default(),
        )
        .expect_err("infinite box must be refused");
    assert!(
        matches!(err, EscalateError::NonFiniteBounds { index: 1, .. }),
        "got {err:?}"
    );
}

/// Scale probe (run manually: `cargo test -p ny-groundtruth --test
/// escalate_smt -- --ignored --nocapture`): where does the exact query stop
/// being decidable in reasonable time? Widens the CROWN-can't case to
/// 6k-neuron single-hidden-layer nets computing the same function
/// (each ±eᵢ row repeated k times with readout weight 2/k), plus the full
/// 3-D box. Results are documented in GEOMETRIC_GROUND_TRUTH_PLAN.md §Route B.
#[test]
#[ignore = "manual scale probe; prints timings"]
fn scale_probe_report_where_ay_times_out() {
    let Some(solver) = solver() else { return };
    let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere builds");

    // Full 3-D box at width 6 first (the baseline hard case).
    let f = abs_net(2.0, -0.75);
    let t0 = std::time::Instant::now();
    let verdict = solver
        .escalate(
            &f,
            &g,
            Relation::Dominates,
            &unit_box(),
            &EscalateOptions {
                timeout_ms: Some(120_000),
                ..EscalateOptions::default()
            },
        )
        .expect("escalation runs");
    println!(
        "width 6 (3-D box): {:?} in {:.1}s",
        verdict_name(&verdict),
        t0.elapsed().as_secs_f64()
    );

    for (k, bias, label) in [
        (2_usize, -0.75, "tight margin 1/4"),
        (4, -0.75, "tight margin 1/4"),
        (8, -0.75, "tight margin 1/4"),
        (16, -0.75, "tight margin 1/4"),
        (2, 1.0, "generous margin 2"),
        (8, 1.0, "generous margin 2"),
        (16, 1.0, "generous margin 2"),
    ] {
        let width = 6 * k;
        let f = wide_abs_net(k, bias);
        let t0 = std::time::Instant::now();
        let verdict = solver
            .escalate(
                &f,
                &g,
                Relation::Dominates,
                &unit_box(),
                &EscalateOptions {
                    timeout_ms: Some(120_000),
                    ..EscalateOptions::default()
                },
            )
            .expect("escalation runs");
        let detail = match &verdict {
            SmtVerdict::Unknown { reason, .. } => format!(" ({reason})"),
            _ => String::new(),
        };
        println!(
            "width {width}, {label}: {:?}{detail} in {:.1}s",
            verdict_name(&verdict),
            t0.elapsed().as_secs_f64()
        );
    }

    // The linear lane (plane g, QF_LRA) for comparison: same nets against
    // g(x) = x2 − 4 (dominated with margin > 1/2 by every family member).
    let plane = signed_plane_distance([0.0, 0.0, 1.0], -4.0).expect("plane builds");
    for k in [2_usize, 8, 16, 64] {
        let width = 6 * k;
        let f = wide_abs_net(k, -0.75);
        let t0 = std::time::Instant::now();
        let verdict = solver
            .escalate(
                &f,
                &plane,
                Relation::Dominates,
                &unit_box(),
                &EscalateOptions {
                    timeout_ms: Some(120_000),
                    ..EscalateOptions::default()
                },
            )
            .expect("escalation runs");
        println!(
            "width {width}, plane g (QF_LRA): {:?} in {:.1}s",
            verdict_name(&verdict),
            t0.elapsed().as_secs_f64()
        );
    }
}

fn verdict_name(v: &SmtVerdict) -> &'static str {
    match v {
        SmtVerdict::Proved { .. } => "Proved",
        SmtVerdict::Falsified { .. } => "Falsified",
        SmtVerdict::ViolationExists { .. } => "ViolationExists",
        SmtVerdict::Unknown { .. } => "Unknown",
        _ => "other",
    }
}

/// The width-6k relative of `abs_net(2, bias)`: every ±eᵢ row appears k
/// times with a *distinct* small hidden bias `j/16384` (all exactly f32), so
/// no two neurons are identical and the encoder's CSE cannot collapse the
/// query. The perturbation shifts f by at most `(2/k)·Σ|b_j| ≤ 72k/16384 <
/// 0.071` for k ≤ 16, so with `bias = −3/4` true dominance survives with
/// margin ≥ 1/4 − 0.071, and with `bias = 1` the margin is generous (≥ 1.9).
fn wide_abs_net(k: usize, bias: f32) -> GraphNetwork {
    let mut rows = Vec::with_capacity(6 * k * 3);
    for _ in 0..k {
        rows.extend_from_slice(&[
            1.0, 0.0, 0.0, //
            -1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, -1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, -1.0,
        ]);
    }
    let w1 = Array2::from_shape_vec((6 * k, 3), rows).expect("shape");
    #[allow(clippy::cast_precision_loss)]
    let b1 = Array1::from_iter((0..6 * k).map(|j| j as f32 / 16384.0));
    #[allow(clippy::cast_precision_loss)]
    let w2 = Array2::from_shape_vec((1, 6 * k), vec![2.0 / k as f32; 6 * k]).expect("shape");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid linear")),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["lin1".to_string()],
    ));
    g.add_node(GraphNode::new(
        "readout",
        Layer::Linear(LinearLayer::new(w2, Some(Array1::from(vec![bias]))).expect("valid linear")),
        vec!["relu".to_string()],
    ));
    g.set_output("readout");
    g
}
