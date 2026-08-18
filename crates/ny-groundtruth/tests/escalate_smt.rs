// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Route B integration tests: SMT escalation against a live `ay` solver
//! (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`).
//!
//! The CROWN-side assertions run in the hermetic default suite. Live SMT
//! conformance is compiled with `--features external-ay` and then requires an
//! `ay` binary at the exact revision pinned by `ny-mip` (`$NY_AY`, `PATH`, or
//! the Trust stage2 sysroot). A selected conformance lane fails on absence or
//! revision drift; it never reports a skipped or vacuous pass.
//!
//! The centrepiece is [`live_ay_proves_crown_unknown_quadratic_dominance`]: a
//! case where the CROWN relaxation of the quadratic ground-truth side is too
//! loose to decide (`PowConstant` secant looseness at the *interior* binding
//! point), but the exact QF_NRA query refutes the violation outright.

use ndarray::{Array1, Array2};
use ny_core::Bound;
#[cfg(feature = "external-ay")]
use ny_groundtruth::SmtVerdict;
use ny_groundtruth::{
    signed_plane_distance, sphere_residual, verify_against_ground_truth, EscalateError,
    EscalateOptions, GroundTruthOutcome, Relation, SmtEscalation,
};
use ny_propagate::layers::{LinearLayer, PowConstantLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
#[cfg(feature = "external-ay")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "external-ay")]
use std::process::Command;

#[cfg(feature = "external-ay")]
fn pinned_ay_revision() -> &'static str {
    let manifest = include_str!("../../ny-mip/Cargo.toml");
    let dependency = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("ay-milp ="))
        .expect("ny-mip must declare ay-milp");
    let revision = dependency
        .split_once("rev = \"")
        .and_then(|(_, tail)| tail.split_once('"'))
        .map(|(revision, _)| revision)
        .expect("ay-milp must remain revision-pinned");
    assert_eq!(revision.len(), 40, "AY revision must be a full commit SHA");
    revision
}

#[cfg(feature = "external-ay")]
fn ay_build_commit(ay: &Path) -> Option<String> {
    let output = Command::new(ay).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("build.commit="))
        .map(str::to_owned)
}

#[cfg(feature = "external-ay")]
fn candidate_ay_paths() -> Vec<PathBuf> {
    if let Some(explicit) = std::env::var_os("NY_AY") {
        return vec![PathBuf::from(explicit)];
    }

    let mut candidates = vec![PathBuf::from("ay")];
    let rustup_trust_ay = std::env::var_os("RUSTUP_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rustup")))
        .map(|root| root.join("toolchains/trust/bin/ay"));
    candidates.extend(rustup_trust_ay);
    candidates
}

/// Require the exact AY revision used by ny-mip. The external conformance lane
/// is explicit, so selecting it without its dependency must fail.
#[cfg(feature = "external-ay")]
fn require_pinned_solver() -> SmtEscalation {
    let expected = pinned_ay_revision();
    let mut observed = Vec::new();
    for candidate in candidate_ay_paths() {
        match ay_build_commit(&candidate) {
            Some(actual) if actual == expected => {
                let workdir =
                    std::env::temp_dir().join(format!("ny-gt-live-ay-{}", std::process::id()));
                std::fs::create_dir_all(&workdir).expect("create live AY test workdir");
                return SmtEscalation::with_solver(candidate, workdir);
            }
            Some(actual) => observed.push(format!("{}: {actual}", candidate.display())),
            None => observed.push(format!(
                "{}: unavailable or no build.commit",
                candidate.display()
            )),
        }
    }
    panic!(
        "live SMT test requires AY revision {expected}; set NY_AY to that binary \
         (observed: {})",
        observed.join(", ")
    );
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

fn plane_case() -> (GraphNetwork, GraphNetwork, Vec<Bound>) {
    (
        abs_net(1.0, 3.0),
        signed_plane_distance([0.0, 0.0, 1.0], -0.5).expect("plane builds"),
        unit_box(),
    )
}

fn quadratic_dominance_case() -> (GraphNetwork, GraphNetwork, Vec<Bound>) {
    (
        abs_net(2.0, -0.75),
        sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere builds"),
        vec![
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
            Bound::new(0.0, 0.0),
        ],
    )
}

#[test]
fn plane_case_crown_verifies() {
    // M0-style case: f = |x0|+|x1|+|x2| + 3 dominates the plane g = x2 − 1/2
    // with margin >= 5/2. This assertion is independent of the live solver.
    let (f, g, bounds) = plane_case();

    let crown =
        verify_against_ground_truth(&f, &g, Relation::Dominates, &bounds).expect("crown path runs");
    assert!(
        matches!(crown, GroundTruthOutcome::Verified { .. }),
        "CROWN must verify the plane case, got {crown:?}"
    );
}

#[test]
#[cfg(feature = "external-ay")]
fn live_ay_plane_case_agrees_with_crown() {
    let (f, g, bounds) = plane_case();
    let crown =
        verify_against_ground_truth(&f, &g, Relation::Dominates, &bounds).expect("crown path runs");
    assert!(
        matches!(crown, GroundTruthOutcome::Verified { .. }),
        "CROWN must verify the plane case, got {crown:?}"
    );

    let solver = require_pinned_solver();
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
fn quadratic_dominance_crown_is_unknown() {
    // f = 2(|x0|+|x1|+|x2|) − 3/4 vs g = ‖x‖² − 1 on [−1,1]² × {0}:
    // h(x) = Σ (2|xi| − xi²) + 1/4, minimum 1/4 > 0 at the box *centre*.
    // CROWN cannot prove it: the lower relaxation of the −xi² terms is the
    // secant (exact only at the endpoints, off by 1 at the interior binding
    // point) and the |xi| lower envelope is linear, so the certified lower
    // bound is ≤ −3/4 + eps.
    let (f, g, bounds) = quadratic_dominance_case();

    let crown =
        verify_against_ground_truth(&f, &g, Relation::Dominates, &bounds).expect("crown path runs");
    assert!(
        matches!(crown, GroundTruthOutcome::Unknown { .. }),
        "CROWN must be too loose here (else the test lost its point), got {crown:?}"
    );
}

#[test]
#[cfg(feature = "external-ay")]
fn live_ay_proves_crown_unknown_quadratic_dominance() {
    // The exact NRA query refutes h < 0 outright after CROWN remains unknown.
    let (f, g, bounds) = quadratic_dominance_case();
    let crown =
        verify_against_ground_truth(&f, &g, Relation::Dominates, &bounds).expect("crown path runs");
    assert!(
        matches!(crown, GroundTruthOutcome::Unknown { .. }),
        "CROWN must be too loose here (else the test lost its point), got {crown:?}"
    );

    let solver = require_pinned_solver();
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
#[cfg(feature = "external-ay")]
fn live_ay_falsified_quadratic_returns_validated_witness() {
    // f = 2(|x0|+|x1|+|x2|) − 5/4 vs the same sphere: h(0) = −1/4 < 0, so
    // dominance is false. The sat model must validate in exact rational
    // arithmetic before being reported.
    let solver = require_pinned_solver();
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
#[cfg(feature = "external-ay")]
fn live_ay_absbound_escalation_proves_self_equivalence() {
    // |f − f| = 0 <= 1/2 everywhere: the AbsBound violation query is unsat.
    let solver = require_pinned_solver();
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
#[cfg(feature = "external-ay")]
fn live_ay_strict_margin_flag_flips_a_touching_case() {
    // f = |x0|+|x1|+|x2| dominates g = plane(x2) − 0 ... f − g = Σ|xi| − x2
    // touches 0 exactly at x = 0. Non-strict dominance holds (unsat);
    // requiring a strict margin makes the violation h <= 0 satisfiable at 0,
    // which validates exactly.
    let solver = require_pinned_solver();
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
    let solver = SmtEscalation::with_solver(PathBuf::from("/nonexistent/ay"), std::env::temp_dir());

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
