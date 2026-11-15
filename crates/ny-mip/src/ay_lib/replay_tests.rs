// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Consumer controls derived from AY's public cert_io fixtures, independently
//! replayed through NY's production adapter. General-integer controls exercise
//! the adapter's model binding; NY's neural IR still admits only ReLU binaries.
use super::*;
use std::time::{Duration, Instant};

const PARITY_INF: &str = "NAME          PARITYINF
ROWS
 N  COST
 E  R0
 E  R1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST      1          R0        1
    X         R1        1
    Y0        R0       -2
    Y1        R1       -2
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       R1       -1
BOUNDS
 BV BND       X
ENDATA
";

const NETWORK_INF: &str = "NAME          NETINF
ROWS
 N  COST
 E  OBJDEF
 E  BAL
 L  VUB
COLUMNS
    FLOW      BAL       1          VUB       1
    OBJ       COST      1          OBJDEF    1
    MARKER                 'MARKER'                 'INTORG'
    ENABLE    OBJDEF   -5          VUB      -1
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       BAL       1
BOUNDS
 FR BND       OBJ
 BV BND       ENABLE
 FX BND       ENABLE    0
ENDATA
";

const DP_INF: &str = "NAME          DPINF
ROWS
 N  COST
 E  R1
COLUMNS
    X         R1        6
    Y         R1        10
    Z         R1        14
RHS
    RHS       R1        18
BOUNDS
 BV BND       X
 BV BND       Y
 BV BND       Z
ENDATA
";

const BDD_INF: &str = "NAME          BDDINF
ROWS
 N  COST
 G  R1
 G  R2
 L  R3
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST      1          R1        2
    X         R3        2
    Y         COST      1          R1        3
    Y         R2        2          R3        2
    Z         COST      1          R2        3
    Z         R3        2
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       R1        4          R2        4
    RHS       R3        5
BOUNDS
 BV BND       X
 BV BND       Y
 BV BND       Z
ENDATA
";

const HYBRID_INF: &str = "NAME          HYBRIDINF
ROWS
 N  COST
 G  RLOW
 L  RUP
COLUMNS
    X         RLOW      -1         RUP       -1
    Y         RLOW       1         RUP        1
RHS
    RHS       RLOW       1         RUP        0
BOUNDS
 BV BND       X
 FR BND       Y
ENDATA
";

const HYBRID_INTEGER_INF: &str = "NAME          HYBRIDINTINF
ROWS
 N  COST
 G  RLOW
 L  RUP
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         RLOW      -1         RUP       -1
    MARKER                 'MARKER'                 'INTEND'
    Y         RLOW       1         RUP        1
RHS
    RHS       RLOW       1         RUP        0
BOUNDS
 UP BND       X           2
 FR BND       Y
ENDATA
";

const OPEN_INF: &str = "NAME          OPENINF
ROWS
 N  COST
 G  ROPEN
 G  RLOW
 L  RUP
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST      1          RLOW      1
    X         RUP       1
    OPEN      COST      1          ROPEN     1
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       ROPEN     2          RLOW      1
    RHS       RUP       0
BOUNDS
 BV BND       X
ENDATA
";

const OPEN_HYBRID_INF: &str = "NAME          OPENHYBRIDINF
ROWS
 N  COST
 G  ROPEN
 G  RLOW
 L  RUP
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST       1         RLOW      -1
    X         RUP       -1
    OPEN      COST       1         ROPEN      1
    MARKER                 'MARKER'                 'INTEND'
    Y         RLOW       1         RUP        1
RHS
    RHS       ROPEN      2         RLOW       1
    RHS       RUP        0
BOUNDS
 BV BND       X
 FR BND       Y
ENDATA
";

const OPEN_HYBRID_INTEGER_INF: &str = "NAME          OPENHYBRIDINTINF
ROWS
 N  COST
 G  ROPEN
 G  RLOW
 L  RUP
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST       1         RLOW      -1
    X         RUP       -1
    OPEN      COST       1         ROPEN      1
    MARKER                 'MARKER'                 'INTEND'
    Y         RLOW       1         RUP        1
RHS
    RHS       ROPEN      2         RLOW       1
    RHS       RUP        0
BOUNDS
 UP BND       X           2
 FR BND       Y
ENDATA
";

fn check_lane(model: Model, lane: &str) {
    let deadline = Instant::now() + Duration::from_secs(20);
    let opts = SolveOpts::new()
        .with_deadline(deadline)
        .with_require_certificates(true)
        .with_engine(EngineEconomics::new().with_hybrid_pb_lp(true));
    let mut session = BabSession::new(model.clone(), &opts).expect("real NY adapter session");
    let outcome = session.check().expect("actual typed-lane solve");
    assert!(
        matches!(outcome, Outcome::Infeasible { .. }),
        "{lane}: {outcome:?}"
    );
    let artifacts = SideArtifactReplays::from_session_with_deadline(&session, Some(deadline));
    assert_eq!(artifacts.admitted_lane, Some(lane), "final admitted lane");
    assert!(
        artifacts.replayable.iter().any(|(name, _)| *name == lane),
        "{lane} artifact missing"
    );
    // Force the positive side-artifact seam even if AY also exported a root
    // or tree proof; an always-rejecting public verifier must fail this test.
    let replayed = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        session.model(),
        &[],
        &[],
        None,
        None,
        Some(deadline),
        artifacts,
        true,
    );
    assert!(
        matches!(replayed, MipResult::Unsat { certified: true }),
        "{lane}: {replayed:?}"
    );
    let mapped = map_bab_outcome(outcome, &session, &[], &[], None, Some(deadline), true);
    assert!(
        matches!(mapped, MipResult::Unsat { certified: true }),
        "{lane}: {mapped:?}"
    );

    let expired = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        session.model(),
        &[],
        &[],
        None,
        None,
        Some(Instant::now()),
        SideArtifactReplays::from_session_with_deadline(&session, Some(Instant::now())),
        true,
    );
    assert!(
        matches!(expired, MipResult::Error(_)),
        "{lane}: expired replay authorized {expired:?}"
    );

    // The same source shape now has no constraints, so the original proof
    // cannot authorize UNSAT. A concrete all-zero source witness is retained.
    let mut relaxed = Model::new();
    let mut objective = Vec::new();
    for index in 0..model.num_cols() {
        let col = model.col_at(index).expect("source column");
        let (lower, upper) = model.col_bounds(col);
        let replacement = match model.col_kind(col) {
            ay_milp::ColKind::Binary => {
                let replacement = relaxed.add_binary_col();
                if lower == upper {
                    relaxed.fix_col(replacement, lower);
                } else {
                    assert_eq!((lower, upper), (0.0, 1.0));
                }
                replacement
            }
            ay_milp::ColKind::Integer => relaxed.add_int_col(lower, upper),
            ay_milp::ColKind::Continuous => relaxed.add_col(lower, upper),
            _ => panic!("unsupported fixture column kind"),
        };
        objective.push((replacement, model.obj_coeff(col)));
    }
    if model.has_objective() {
        relaxed.set_objective(&objective, model.sense());
        relaxed.set_objective_offset(model.objective_offset());
    }
    for _ in 0..model.num_rows() {
        relaxed.add_row(f64::NEG_INFINITY, f64::INFINITY, &[]);
    }
    relaxed
        .check_point(&vec![BigRational::from_integer(0.into()); model.num_cols()])
        .expect("relaxed source really is feasible");
    let mutated = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        &relaxed,
        &[],
        &[],
        None,
        None,
        None,
        SideArtifactReplays::from_session(&session),
        true,
    );
    assert!(
        matches!(mutated, MipResult::Error(_)),
        "{lane}: wrong source authorized {mutated:?}"
    );
}

fn bare_infeasible() -> Outcome {
    Outcome::Infeasible {
        cert: None,
        tree_cert: None,
    }
}

#[test]
fn public_replay_parity_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(PARITY_INF).expect("fixture").model,
        "parity",
    );
}

#[test]
fn public_replay_network_design_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(NETWORK_INF).expect("fixture").model,
        "network_design",
    );
}

#[test]
fn public_replay_single_row_dp_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(DP_INF).expect("fixture").model,
        "single_row_dp",
    );
}

#[test]
fn public_replay_multi_row_bdd_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(BDD_INF).expect("fixture").model,
        "multi_row_bdd",
    );
}

#[test]
fn public_replay_hybrid_pb_lp_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(HYBRID_INF).expect("fixture").model,
        "hybrid_pb_lp",
    );
}

#[test]
fn public_replay_hybrid_integer_lift_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(HYBRID_INTEGER_INF)
            .expect("fixture")
            .model,
        "hybrid_integer_lift",
    );
}

#[test]
fn public_replay_open_domain_single_row_dp_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(OPEN_INF).expect("fixture").model,
        "open_domain_single_row_dp",
    );
}

#[test]
fn public_replay_open_domain_hybrid_pb_lp_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(OPEN_HYBRID_INF).expect("fixture").model,
        "open_domain_hybrid_pb_lp",
    );
}

#[test]
fn public_replay_open_domain_hybrid_integer_lift_binds_source_and_original_deadline() {
    check_lane(
        ay_milp::read_mps(OPEN_HYBRID_INTEGER_INF)
            .expect("fixture")
            .model,
        "open_domain_hybrid_integer_lift",
    );
}

#[test]
fn public_replay_open_domain_multi_row_bdd_binds_source_and_original_deadline() {
    let mut model = ay_milp::read_mps(BDD_INF).expect("BDD fixture").model;
    let open = model.add_int_col(0.0, f64::INFINITY);
    model.add_row(2.0, f64::INFINITY, &[(open, 1.0)]);
    check_lane(model, "open_domain_multi_row_bdd");
}

#[test]
fn advisory_attribution_alone_does_not_certify_a_verdict() {
    let artifacts = SideArtifactReplays {
        admitted_lane: Some("hybrid_integer_lift"),
        ..SideArtifactReplays::none()
    };
    let mapped = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        &Model::new(),
        &[],
        &[],
        None,
        None,
        None,
        artifacts,
        true,
    );
    assert!(matches!(mapped, MipResult::Unsat { certified: false }));
}

fn affine_problem(second_rhs: f64) -> MilpProblem {
    let mut problem = MilpProblem::new();
    let z = problem.add_integer_col(0.0, 0.0, 0.0);
    let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
    let y = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
    problem.add_row(0.0, 0.0, [(z, 1.0)]);
    problem.add_row(0.0, 0.0, [(x, 1.0), (y, 1.0)]);
    problem.add_row(second_rhs, second_rhs, [(x, 1.0), (y, 1.0)]);
    problem
}

fn affine_options(deadline: Instant) -> SolveOpts {
    SolveOpts::new()
        .with_deadline(deadline)
        .with_structure_routing(false)
        .with_require_certificates(true)
        .with_tree_cert_leaves(64)
        .with_engine(
            EngineEconomics::new()
                .with_affine_agg(true)
                .with_presolve(false)
                .with_lattice(false)
                .with_root_probe(false)
                .with_kernel_reformulation(false)
                .with_binary_complement_sub(false)
                .with_dual_fixing(false)
                .with_dedup_cols(false),
        )
}

#[test]
fn public_replay_affine_actual_ny_source_finalizer_mutation_and_deadline() {
    let source = to_ay_model(&affine_problem(1.0)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut session = BabSession::new(source, &affine_options(deadline)).unwrap();
    let outcome = session.check().unwrap();
    assert!(matches!(outcome, Outcome::Infeasible { .. }), "{outcome:?}");
    let certificate = session
        .affine_aggregation_certificate()
        .expect("actual NY-lowered affine artifact");
    assert!(
        certificate
            .verify(session.model())
            .unwrap()
            .infeasibility_verified
    );
    let wrong_frame_rejected = match certificate.inner_proof() {
        ay_milp::AffineAggregationInnerProof::Farkas(proof) => {
            proof.verify(session.model()).is_err()
        }
        ay_milp::AffineAggregationInnerProof::InfeasibilityTree(proof) => {
            proof.verify(session.model()).is_err()
        }
        _ => panic!("actual checked affine infeasibility needs an inner refutation"),
    };
    assert!(
        wrong_frame_rejected,
        "inner proof must not inherit source-frame authority"
    );
    let artifacts = SideArtifactReplays::from_session_with_deadline(&session, Some(deadline));
    assert_eq!(artifacts.admitted_lane, Some("affine_aggregation"));
    assert_eq!(
        artifacts
            .replayable
            .iter()
            .map(|(lane, _)| *lane)
            .collect::<Vec<_>>(),
        ["affine_aggregation"]
    );
    // Force side replay rather than letting the ordinary source tree hide an
    // affine-adapter failure. Both proofs came from the actual session above.
    let mapped = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        session.model(),
        &[],
        &[],
        None,
        None,
        Some(deadline),
        artifacts,
        true,
    );
    assert!(
        matches!(mapped, MipResult::Unsat { certified: true }),
        "{mapped:?}"
    );
    assert!(matches!(
        map_bab_outcome(outcome, &session, &[], &[], None, Some(deadline), true),
        MipResult::Unsat { certified: true }
    ));

    let feasible = to_ay_model(&affine_problem(0.0)).unwrap();
    feasible
        .check_point(&vec![BigRational::from_integer(0.into()); 3])
        .unwrap();
    assert!(replay_affine_infeasibility(&feasible, certificate, Some(deadline)).is_err());
    let mut relaxed_kind = affine_problem(1.0);
    relaxed_kind.relax_integrality();
    assert!(replay_affine_infeasibility(
        &to_ay_model(&relaxed_kind).unwrap(),
        certificate,
        Some(deadline)
    )
    .is_err());
    let expired = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        session.model(),
        &[],
        &[],
        None,
        None,
        Some(Instant::now()),
        SideArtifactReplays::from_session_with_deadline(&session, Some(Instant::now())),
        true,
    );
    assert!(matches!(expired, MipResult::Error(_)), "{expired:?}");
    assert!(Instant::now() < deadline);
}

#[test]
fn public_replay_affine_actual_unsupported_infeasibility_never_certifies() {
    let mut source = Model::new();
    let x = source.add_int_col(0.0, 5.0);
    let y = source.add_int_col(0.0, 5.0);
    source.add_row(0.0, 0.0, &[(x, 1.0), (y, -1.0)]);
    source.add_row(1.0, 1.0, &[(x, 1.0), (y, 1.0)]);
    let deadline = Instant::now() + Duration::from_secs(20);
    let opts = affine_options(deadline)
        .with_require_certificates(false)
        .with_tree_cert_leaves(0);
    let mut session = BabSession::new(source, &opts).unwrap();
    let outcome = session.check().unwrap();
    assert!(matches!(outcome, Outcome::Infeasible { .. }), "{outcome:?}");
    let certificate = session
        .affine_aggregation_certificate()
        .expect("actual unsupported affine artifact");
    assert_eq!(
        certificate.claim(),
        &ay_milp::AffineAggregationClaim::Infeasible
    );
    assert!(matches!(
        certificate.inner_proof(),
        ay_milp::AffineAggregationInnerProof::Unsupported
    ));
    assert!(
        !certificate
            .verify(session.model())
            .unwrap()
            .infeasibility_verified
    );
    assert_eq!(session.admitted_infeasibility_evidence(), None);
    let mapped = map_outcome_with_sat_relu_certificate(
        bare_infeasible(),
        session.model(),
        &[],
        &[],
        None,
        None,
        Some(deadline),
        SideArtifactReplays::from_session_with_deadline(&session, Some(deadline)),
        false,
    );
    assert!(matches!(mapped, MipResult::Error(_)), "{mapped:?}");
}
