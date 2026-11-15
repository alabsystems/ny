// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ay_milp::Multiplier;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(20)
}

fn constant_difference() -> SourceModel {
    SourceModel::parse(
        "PAIRED_LP_GEQ_V1 3 2 0 1\n0 1 0.25 1.25 0 1\n1 2 1 1 0 -1 0.25\n2 2 2 1 0 -1 0\n1 2\n",
    )
    .unwrap()
}

fn cert(prepared: &Prepared, source: &SourceModel, sign: f64) -> OptimalityCertificate {
    OptimalityCertificate {
        sense: Sense::Minimize,
        objective: objective(source, 0, sign),
        bound: rational(sign * 0.25),
        multipliers: vec![
            Multiplier {
                fact: FactRef::RowBound {
                    row: prepared.model.row_at(0).unwrap(),
                    side: if sign > 0.0 {
                        BoundSide::Lower
                    } else {
                        BoundSide::Upper
                    },
                },
                coeff: rational(1.0),
            },
            Multiplier {
                fact: FactRef::RowBound {
                    row: prepared.model.row_at(1).unwrap(),
                    side: if sign > 0.0 {
                        BoundSide::Upper
                    } else {
                        BoundSide::Lower
                    },
                },
                coeff: rational(1.0),
            },
        ],
    }
    .tap_verify(&prepared.model)
}

trait VerifyTap {
    fn tap_verify(self, model: &Model) -> Self;
}
impl VerifyTap for OptimalityCertificate {
    fn tap_verify(self, model: &Model) -> Self {
        self.verify(model).unwrap();
        self
    }
}

#[test]
fn equality_lower_and_upper_map_to_correct_free_dual_signs() {
    let source = constant_difference();
    let prepared = prepare(&source, deadline()).unwrap();
    for sign in [1.0, -1.0] {
        let artifact = cert(&prepared, &source, sign);
        let dual = extract(&artifact, &prepared, &source, deadline()).unwrap();
        assert_eq!(dual.equalities, vec![sign, -sign]);
        assert_eq!(
            exact_replay(&source, &dual, 0, sign, deadline()).unwrap(),
            rational(sign * 0.25)
        );
    }
}

#[test]
fn geq_lower_maps_positive_and_forged_upper_is_rejected() {
    let source =
        SourceModel::parse("PAIRED_LP_GEQ_V1 2 1 1 1\n0 1 0 0\n1 1 1 1 0\n1 0 1 0.25\n0 1\n")
            .unwrap();
    let prepared = prepare(&source, deadline()).unwrap();
    let artifact = OptimalityCertificate {
        sense: Sense::Minimize,
        objective: objective(&source, 0, 1.0),
        bound: rational(0.25),
        multipliers: vec![
            Multiplier {
                fact: FactRef::RowBound {
                    row: prepared.model.row_at(1).unwrap(),
                    side: BoundSide::Lower,
                },
                coeff: rational(1.0),
            },
            Multiplier {
                fact: FactRef::RowBound {
                    row: prepared.model.row_at(0).unwrap(),
                    side: BoundSide::Upper,
                },
                coeff: rational(1.0),
            },
        ],
    };
    artifact.verify(&prepared.model).unwrap();
    let dual = extract(&artifact, &prepared, &source, deadline()).unwrap();
    assert_eq!(dual.inequalities, vec![1.0]);
    assert_eq!(dual.equalities, vec![-1.0]);
    let mut forged = artifact;
    forged.multipliers[0].fact = FactRef::RowBound {
        row: prepared.model.row_at(1).unwrap(),
        side: BoundSide::Upper,
    };
    assert!(forged.verify(&prepared.model).is_err());
    assert!(extract(&forged, &prepared, &source, deadline()).is_err());
}

#[test]
fn nonrepresentable_duplicate_sum_is_lifted_without_changing_exact_primal() {
    let source = SourceModel::parse("PAIRED_LP_GEQ_V1 3 2 0 1\n0 1 0 18014398509481984 0 1\n1 3 1 1 0 -9007199254740992 0 -1 0\n2 2 2 1 0 -1 0\n1 2\n").unwrap();
    let prepared = prepare(&source, deadline()).unwrap();
    assert_eq!(prepared.aliases, 1);
    assert_eq!(prepared.cols.len(), 4);
    let y = BigRational::from_integer(9_007_199_254_740_993_i64.into());
    let point = vec![rational(1.0), y, rational(1.0), rational(1.0)];
    prepared.model.check_point(&point).unwrap();
    check_original_primal(&source, &point[..3], deadline()).unwrap();
    let mut forged = point;
    forged[1] = rational(9_007_199_254_740_992.0);
    assert!(prepared.model.check_point(&forged).is_err());
    assert!(check_original_primal(&source, &forged[..3], deadline()).is_err());
}

#[test]
fn exact_duplicate_cancellation_stays_one_source_column() {
    let source =
        SourceModel::parse("PAIRED_LP_GEQ_V1 2 1 0 1\n-1 1 0 0\n1 3 1 1 0 1 0 -1 0\n0 1\n")
            .unwrap();
    let prepared = prepare(&source, deadline()).unwrap();
    assert_eq!(prepared.aliases, 0);
    assert_eq!(
        prepared.model.row(prepared.model.row_at(0).unwrap()).0,
        &[(1, 1.0)]
    );
}

#[test]
fn strict_lp_admission_refuses_forged_sense_shapes_and_numbers() {
    let good = "PAIRED_LP_GEQ_V1 2 1 0 1\n0 1 0 0\n1 1 1 1 0\n0 1\n";
    assert!(SourceModel::parse(good).is_ok());
    for bad in [
        good.replace("GEQ", "LEQ"),
        good.replace("0 1 0 0", "NaN 1 0 0"),
        good.replace("1 1 1 1 0", "1 1 1 -1 0"),
        good.to_owned() + " changed_rhs",
    ] {
        assert!(SourceModel::parse(&bad).is_err());
    }
}

#[test]
fn ay_primal_fraction_is_verified_exactly_and_rounding_is_diagnostic() {
    let source = SourceModel::parse(
        "PAIRED_LP_GEQ_V1 3 2 1 1\n0 1 0 3 0 0\n1 2 1 1 0 -3 0\n2 1 2 1 0\n1 1 1 1\n0 2\n",
    )
    .unwrap();
    let result = solve(&source, deadline(), 128 * 1024 * 1024).unwrap();
    assert_eq!(result.finished_objectives, 2, "{:?}", result.diagnostics);
    assert_eq!(
        result.diagnostics[0].ay_exact_optimum.as_deref(),
        Some("1/3")
    );
    assert_eq!(result.diagnostics[0].primal_source_values[0], "1/3");
    assert!(result.diagnostics[0].exact_primal_checked);
}

#[test]
fn expired_solve_cannot_publish_new_artifact() {
    let source = constant_difference();
    assert!(solve(
        &source,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        128 * 1024 * 1024
    )
    .is_err());
    assert!(solve(&source, deadline(), usize::MAX).is_err());
    assert!(solve_float(
        &source,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        128 * 1024 * 1024
    )
    .is_err());
}

#[test]
fn floating_proposals_retain_both_conventions_and_claim_no_exact_primal() {
    let source = constant_difference();
    let result = solve_float(&source, deadline(), 128 * 1024 * 1024).unwrap();
    assert_eq!(result.finished_objectives, 2, "{:?}", result.diagnostics);
    assert!(result.alternate_dual_text.is_some());
    for d in &result.diagnostics {
        assert_eq!(d.status, "untrusted_float_duals");
        assert!(!d.exact_primal_checked);
        assert!(d.ay_exact_optimum.is_none());
        assert_eq!(d.primal_source_values, [] as [String; 0]);
        assert!(d.numerical_proposal_lower.unwrap() >= -0.251);
    }
}

#[test]
fn wrong_signed_float_geq_is_dropped_and_full_residual_is_priced() {
    let source =
        SourceModel::parse("PAIRED_LP_GEQ_V1 2 1 1 1\n0 1 0 0\n1 1 1 1 0\n1 0 1 0.25\n0 1\n")
            .unwrap();
    let prepared = prepare(&source, deadline()).unwrap();
    let first = extract_float(&[-1.0, 1.0], &prepared, &source, 1.0).unwrap();
    let other = extract_float(&[-1.0, 1.0], &prepared, &source, -1.0).unwrap();
    assert_eq!(first.inequalities, vec![1.0]);
    assert_eq!(other.inequalities, vec![0.0]);
    assert_eq!(
        exact_replay(&source, &first, 0, 1.0, deadline()).unwrap(),
        rational(0.25)
    );
    assert_eq!(
        exact_replay(&source, &other, 0, 1.0, deadline()).unwrap(),
        rational(0.0)
    );
    assert!(extract_float(&[f64::NAN, 1.0], &prepared, &source, 1.0).is_err());
    assert!(extract_float(&[1.0], &prepared, &source, 1.0).is_err());
}

#[test]
fn worker_permit_is_busy_until_actual_worker_exit_and_releases_on_panic() {
    let counter = AtomicUsize::new(0);
    let first = acquire_worker(&counter, 2).unwrap();
    let second = acquire_worker(&counter, 2).unwrap();
    assert!(acquire_worker(&counter, 2).is_err());
    // Caller-side objects disappearing do not drop a worker-owned permit.
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let (exit_tx, exit_rx) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let _permit = second;
            let _ = exit_rx.recv();
            let _ = done_tx.send(());
        });
        assert_eq!(counter.load(Ordering::Acquire), 2);
        assert!(acquire_worker(&counter, 2).is_err());
        exit_tx.send(()).unwrap();
        done_rx.recv().unwrap();
    });
    assert_eq!(counter.load(Ordering::Acquire), 1);
    drop(first);
    assert_eq!(counter.load(Ordering::Acquire), 0);
    let _ = std::panic::catch_unwind(|| {
        let _permit = acquire_worker(&counter, 2).unwrap();
        panic!("test worker panic");
    });
    assert_eq!(counter.load(Ordering::Acquire), 0);
}
