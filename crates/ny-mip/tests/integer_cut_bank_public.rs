// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! A consumer uses only public AY and NY APIs to transfer actual finished cuts.

use std::time::{Duration, Instant};

use ay_milp::{BabSession, EngineEconomics, Model, Outcome, Sense, SolveOpts};
use num_rational::BigRational;
use ny_mip::ir::Col;
use ny_mip::{
    IntegerCutBank, MilpProblem, MipBackend, MipConfig, MipFeasibilityIngress, MipParts, MipResult,
    MipSolver,
};

fn sources(upper: f64) -> (MilpProblem, Model) {
    let mut problem = MilpProblem::new();
    let x = problem.add_integer_col(0.0, 0.0, 1.0);
    let y = problem.add_integer_col(0.0, 0.0, 1.0);
    problem.add_row(f64::NEG_INFINITY, upper, [(x, 2.0), (y, 2.0)]);

    // Independent public construction, without NY's private lowering helper.
    let mut model = Model::new();
    let x = model.add_binary_col();
    let y = model.add_binary_col();
    model.add_row(f64::NEG_INFINITY, upper, &[(x, 2.0), (y, 2.0)]);
    model.set_objective(&[(x, 1.0), (y, 1.0)], Sense::Maximize);
    (problem, model)
}

fn finished(model: Model, deadline: Instant) -> BabSession {
    let options = SolveOpts::new()
        .with_deadline(deadline)
        .with_time_limit(Duration::from_secs(20))
        .with_structure_routing(false)
        .with_engine(
            EngineEconomics::new()
                .with_lattice(false)
                .with_dedup_cols(false)
                .with_presolve(false)
                .with_root_probe(false)
                .with_sym(false)
                .with_coef_tighten(false)
                .with_cuts(true),
        );
    let mut session = BabSession::new(model, &options).unwrap();
    let outcome = session.check().unwrap();
    assert!(matches!(outcome, Outcome::Optimal { .. }), "{outcome:?}");
    let coverage = session.cut_harvest_summary();
    assert!(coverage.native_tree_finished);
    assert!(coverage.adoption_scan_complete);
    assert!(coverage.certified_sides > 0);
    session
}

fn solver(problem: MilpProblem) -> MipSolver {
    let num_cols = problem.num_cols();
    MipSolver::new(
        MipParts {
            problem,
            input_vars: vec![Col(0), Col(1)],
            output_vars: Vec::new(),
            binary_vars: Vec::new(),
            binary_widths: Vec::new(),
            num_cols,
        },
        MipConfig {
            backend: MipBackend::Ay,
            feasibility_ingress: MipFeasibilityIngress::Historical,
            parallel_split: 1,
            timeout_secs: 5.0,
            ..MipConfig::default()
        },
    )
}

#[test]
fn public_finished_cut_intake_replays_siblings_source_deadline_and_lifecycle() {
    let deadline = Instant::now() + Duration::from_secs(20);
    let (problem, model) = sources(3.0);
    let mut session = finished(model, deadline);
    let mut bank = IntegerCutBank::default();
    let report = bank
        .harvest_from_finished_session(&mut session, &problem, deadline)
        .unwrap();
    assert!(report.received_sides > 0);
    assert!(report.added > 0);
    let retained = bank.len();
    assert!(retained > 0, "an actual cut must cross the public seam");

    let drained = bank
        .harvest_from_finished_session(&mut session, &problem, deadline)
        .unwrap();
    assert_eq!(drained.received_sides, 0);
    assert_eq!(drained.added, 0);
    assert_eq!(drained.certified_sides, report.certified_sides);
    assert_eq!(bank.len(), retained);

    for fixed in [0.0, 1.0] {
        let mut sibling = problem.clone();
        sibling.fix_col(Col(0), fixed);
        let mut sibling_bank = bank.clone();
        let result = solver(sibling)
            .check_feasibility_with_integer_cut_bank(&mut sibling_bank)
            .unwrap();
        assert!(matches!(result, MipResult::Sat { .. }), "{result:?}");
        assert_eq!(sibling_bank.len(), retained);
    }

    let (changed, changed_model) = sources(4.0);
    changed_model
        .check_point(&[
            BigRational::from_integer(1.into()),
            BigRational::from_integer(1.into()),
        ])
        .unwrap();
    assert!(solver(changed.clone())
        .check_feasibility_with_integer_cut_bank(&mut bank)
        .is_err());
    assert_eq!(bank.len(), retained);

    let (_, model) = sources(3.0);
    let mut foreign = finished(model, deadline);
    let mut refused = IntegerCutBank::default();
    let omitted = refused
        .harvest_from_finished_session(&mut foreign, &changed, deadline)
        .unwrap();
    assert!(omitted.received_sides > 0);
    assert_eq!(omitted.source_incompatible, omitted.received_sides);
    assert_eq!(omitted.added, 0);
    assert_eq!(refused.len(), 0);

    let (_, model) = sources(3.0);
    let mut expired = finished(model, deadline);
    assert!(bank
        .harvest_from_finished_session(&mut expired, &problem, Instant::now())
        .is_err());
    assert_eq!(bank.len(), retained);
    assert_ne!(expired.harvest_cuts().len(), 0, "expiry precedes drain");

    let (_, model) = sources(3.0);
    let mut changed_session = finished(model, deadline);
    let column = changed_session.model().col_at(0).unwrap();
    changed_session.fix_col(column, 0.0).unwrap();
    assert!(bank
        .harvest_from_finished_session(&mut changed_session, &problem, deadline)
        .is_err());
    assert_eq!(bank.len(), retained);
}
