// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::encoder::MipParts;
use crate::{ir::Col, MilpProblem, MipConfig, MipResult, MipSolver};
use ay_milp::{EngineEconomics, LpSession, Outcome, Sense, SolveOpts};
use std::time::Duration;

fn source(upper: f64) -> MilpProblem {
    let mut problem = MilpProblem::new();
    let x = problem.add_integer_col(0.0, 0.0, 1.0);
    let y = problem.add_integer_col(0.0, 0.0, 1.0);
    problem.add_row(f64::NEG_INFINITY, upper, [(x, 2.0), (y, 2.0)]);
    problem
}

fn actual_native_bank() -> IntegerCutBank {
    actual_native_bank_for(&source(3.0))
}

fn actual_native_bank_for(problem: &MilpProblem) -> IntegerCutBank {
    let (mut session, deadline) = actual_native_session_for(problem);
    let mut bank = IntegerCutBank::default();
    let report = bank
        .harvest_from_finished_session(&mut session, problem, deadline)
        .unwrap();
    assert!(report.received_sides > 0);
    assert!(bank.last_report().native_tree_finished);
    assert!(bank.last_report().adoption_scan_complete);
    assert!(bank.last_report().adopted_root_sides > 0);
    assert!(bank.last_report().added > 0);
    assert_ne!(
        bank.len(),
        0,
        "an actual retained cut, never a shape-only test"
    );
    bank
}

fn actual_native_session_for(problem: &MilpProblem) -> (BabSession, Instant) {
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut model = crate::ay_lib::to_ay_model(problem).unwrap();
    model.set_objective(
        &[
            (model.col_at(0).unwrap(), 1.0),
            (model.col_at(1).unwrap(), 1.0),
        ],
        Sense::Maximize,
    );
    let opts = SolveOpts::new()
        .with_deadline(deadline)
        .with_structure_routing(false)
        .with_time_limit(Duration::from_secs(20))
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
    let mut session = BabSession::new(model, &opts).unwrap();
    let outcome = session.check().unwrap();
    assert!(matches!(outcome, Outcome::Optimal { .. }), "{outcome:?}");
    assert!(session.cut_harvest_summary().certified_sides > 0);
    (session, deadline)
}

fn solver(problem: MilpProblem) -> MipSolver {
    solver_with_timeout(problem, 5.0)
}

fn solver_with_timeout(problem: MilpProblem, timeout_secs: f64) -> MipSolver {
    let num_cols = problem.num_cols();
    MipSolver::new(
        MipParts {
            problem,
            input_vars: vec![Col(0), Col(1)],
            output_vars: vec![],
            binary_vars: vec![],
            binary_widths: vec![],
            num_cols,
        },
        MipConfig {
            parallel_split: 1,
            timeout_secs,
            ..MipConfig::default()
        },
    )
}

#[test]
fn integer_cut_intake_deadline_expires_inside_one_wide_original_source_row() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_POLLS_LEFT.with(|remaining| remaining.set(None));
        }
    }
    let _reset = Reset;
    let small = source(3.0);
    let (mut session, deadline) = actual_native_session_for(&small);
    let mut accepted = IntegerCutBank::default();
    TEST_POLLS_LEFT.with(|remaining| remaining.set(Some(64)));
    let report = accepted
        .harvest_from_finished_session(&mut session, &small, deadline)
        .expect("same work budget admits actual finished evidence for the small source");
    assert!(report.added > 0);
    TEST_POLLS_LEFT.with(|remaining| remaining.set(None));

    let mut wide = small.clone();
    // This additional row is redundant with binary x<=1. The first source row
    // and its genuine native cut remain unchanged. A single source coefficient
    // scan, rather than cut verification or many source rows, consumes budget.
    const WIDTH: usize = 65_536;
    wide.add_row(
        f64::NEG_INFINITY,
        WIDTH as f64,
        (0..WIDTH).map(|_| (Col(0), 1.0)),
    );
    let (mut session, deadline) = actual_native_session_for(&small);
    let mut rejected = IntegerCutBank::default();
    TEST_POLLS_LEFT.with(|remaining| remaining.set(Some(64)));
    let error = rejected
        .harvest_from_finished_session(&mut session, &wide, deadline)
        .expect_err("source reconstruction must cooperate before the row is copied");
    assert!(error.to_string().contains("deadline expired"));
    assert!(rejected.is_empty());
    TEST_POLLS_LEFT.with(|remaining| remaining.set(None));
    assert_ne!(session.harvest_cuts().len(), 0, "expiry precedes drain");
    let (mut session, deadline) = actual_native_session_for(&small);
    let completed = rejected
        .harvest_from_finished_session(&mut session, &wide, deadline)
        .expect("the wide redundant source is valid when work budget permits");
    assert!(completed.added > 0);
}

#[test]
fn integer_cut_bank_reuses_real_finished_cut_across_both_sibling_domains() {
    let bank = actual_native_bank();
    for fixed in [0.0, 1.0] {
        let mut child = source(3.0);
        child.fix_col(Col(0), fixed);
        let mut sibling_bank = bank.clone();
        let result = solver(child)
            .check_feasibility_with_integer_cut_bank(&mut sibling_bank)
            .unwrap();
        assert!(matches!(result, MipResult::Sat { .. }), "{result:?}");
        assert_eq!(sibling_bank.len(), bank.len());
    }
    let mut cleared = bank;
    cleared.clear();
    assert_eq!(cleared.len(), 0);
    assert_eq!(cleared.last_report(), IntegerCutBankReport::default());
}

#[test]
fn integer_cut_bank_composes_integer_premises_with_augmented_farkas_only() {
    let mut bank = actual_native_bank();
    let mut source = source(3.0);
    // The original LP point (1/2,1) satisfies both rows. Only integer rounding
    // licenses x+y<=1; a root Farkas over that cut cannot prove the source LP.
    source.add_row(1.5, f64::INFINITY, [(Col(0), 1.0), (Col(1), 1.0)]);
    let mut relaxation = source.clone();
    relaxation.relax_integrality();
    let source_lp = crate::ay_lib::to_ay_model(&relaxation).unwrap();
    source_lp
        .check_point(&[
            BigRational::new(1.into(), 2.into()),
            BigRational::from_integer(1.into()),
        ])
        .unwrap();
    let source_model = crate::ay_lib::to_ay_model(&source).unwrap();
    let augmented = bank.augment(&source_model, None).unwrap();
    // LpSession intentionally rejects integer models. Copy the already
    // licensed augmented rows into a continuous test-only frame, retaining
    // the exact f64 facts and row order for Farkas replay below.
    let mut augmented_lp = Model::new();
    for index in 0..augmented.num_cols() {
        let bounds = augmented.col_bounds(augmented.col_at(index).unwrap());
        augmented_lp.add_col(bounds.0, bounds.1);
    }
    for index in 0..augmented.num_rows() {
        let (entries, lower, upper) = augmented.row(augmented.row_at(index).unwrap());
        let entries: Vec<_> = entries
            .iter()
            .map(|(column, a)| (augmented_lp.col_at(*column as usize).unwrap(), *a))
            .collect();
        augmented_lp.add_row(lower, upper, &entries);
    }
    let opts = SolveOpts::new().with_time_limit(Duration::from_secs(5));
    let mut lp = LpSession::new(&augmented_lp, &opts).unwrap();
    let outcome = lp.optimize_model_objective().unwrap();
    let Outcome::Infeasible {
        cert: Some(cert), ..
    } = outcome
    else {
        panic!("the cut must enable an actual root Farkas: {outcome:?}");
    };
    cert.verify(&augmented).unwrap();
    assert!(
        cert.verify(&source_model).is_err(),
        "augmented LP authority cannot escape its frame"
    );
    assert!(
        bank.augment(&source_lp, None).is_err(),
        "relaxed integrality loses cut premises"
    );
    let result = solver(source)
        .check_feasibility_with_integer_cut_bank(&mut bank)
        .unwrap();
    assert!(
        matches!(result, MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

#[test]
fn integer_cut_bank_changed_source_deadline_and_nonrepresentable_rows_refuse() {
    let mut bank = actual_native_bank();
    let retained = bank.len();
    let relaxed = source(4.0);
    let model = crate::ay_lib::to_ay_model(&relaxed).unwrap();
    model
        .check_point(&vec![BigRational::from_integer(1.into()); 2])
        .unwrap();
    assert!(solver(relaxed)
        .check_feasibility_with_integer_cut_bank(&mut bank)
        .is_err());
    assert_eq!(
        bank.len(),
        retained,
        "failed recipient does not replace the bank"
    );
    assert!(solver_with_timeout(source(3.0), f64::NAN)
        .check_feasibility_with_integer_cut_bank(&mut bank)
        .is_err());
    assert_eq!(bank.len(), retained);
    let original = crate::ay_lib::to_ay_model(&source(3.0)).unwrap();
    assert!(bank.augment(&original, Some(Instant::now())).is_err());
    assert!(bank.snapshot_with_deadline(Instant::now()).is_none());
    let snapshot = bank
        .snapshot_with_deadline(Instant::now() + Duration::from_secs(5))
        .expect("in-budget bank snapshot");
    assert_eq!(snapshot.cuts, bank.cuts);
    assert_eq!(snapshot.last_report(), bank.last_report());
    assert_eq!(bank.len(), retained);

    assert!(exact_float(&BigRational::new(1.into(), 3.into())).is_err());
    assert_eq!(
        exact_float(&BigRational::new(1.into(), 2.into())).unwrap(),
        0.5
    );
    let mut late_solver = solver(source(3.0));
    late_solver
        .set_ay_hard_deadline(5.0, Instant::now())
        .unwrap();
    let result = late_solver
        .check_feasibility_with_integer_cut_bank(&mut bank)
        .unwrap();
    assert!(matches!(result, MipResult::Timeout));
    assert_eq!(bank.len(), retained);
}

#[test]
fn integer_cut_bank_cannot_transport_domain_local_source_rows() {
    let mut local = source(4.0);
    local.add_row(f64::NEG_INFINITY, 3.0, [(Col(0), 2.0), (Col(1), 2.0)]);
    let bank = actual_native_bank_for(&local);
    // The actual native derivation uses the stronger row present only here.
    bank.augment(&crate::ay_lib::to_ay_model(&local).unwrap(), None)
        .unwrap();
    assert!(bank
        .augment(&crate::ay_lib::to_ay_model(&source(4.0)).unwrap(), None)
        .is_err());
}

#[test]
fn integer_cut_bank_snapshot_polls_inside_one_source_valid_row() {
    // A mathematically verified work fixture, not a claim of native cut origin.
    // Actual native provenance has separate positive controls above and through
    // the external public API. This isolates the cooperative coefficient copy.
    const WIDTH: usize = 16_384;
    let mut model = Model::new();
    let columns: Vec<_> = (0..WIDTH).map(|_| model.add_binary_col()).collect();
    let entries: Vec<_> = columns.iter().map(|&column| (column, 2.0)).collect();
    let row = model.add_row(f64::NEG_INFINITY, 3.0, &entries);
    let cut = CertifiedIntegerCut {
        coeffs: (0..WIDTH)
            .map(|column| {
                (
                    u32::try_from(column).expect("fixture column fits"),
                    BigRational::from_integer((-1).into()),
                )
            })
            .collect(),
        lb: BigRational::from_integer((-1).into()),
        proof: ay_milp::IntegerCutProof::SourceRowRounding {
            row,
            side: ay_milp::BoundSide::Upper,
            multiplier: BigRational::new(1.into(), 2.into()),
            unrounded_bound: BigRational::new((-3).into(), 2.into()),
            output_scale: BigRational::from_integer(1.into()),
        },
    };
    cut.verify(&model)
        .expect("source-valid integer rounding fixture");
    let large = IntegerCutBank {
        cuts: vec![cut],
        ..IntegerCutBank::default()
    };
    let actual = actual_native_bank();
    let small = IntegerCutBank {
        cuts: vec![actual.cuts[0].clone()],
        ..IntegerCutBank::default()
    };
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_POLLS_LEFT.with(|remaining| remaining.set(None));
        }
    }
    let _reset = Reset;
    let deadline = Instant::now() + Duration::from_secs(5);
    TEST_POLLS_LEFT.with(|remaining| remaining.set(Some(16)));
    assert!(small.snapshot_with_deadline(deadline).is_some());
    TEST_POLLS_LEFT.with(|remaining| remaining.set(Some(16)));
    assert!(large.snapshot_with_deadline(deadline).is_none());
    assert_eq!(large.cuts[0].coeffs.len(), WIDTH);
}
