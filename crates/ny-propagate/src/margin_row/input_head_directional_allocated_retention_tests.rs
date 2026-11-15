// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Shared debt/resource/retention controls; provider hints below are mocks.
use super::allocated_tests::*;
use super::*;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn allocated_discovery_rejects_overlapping_negative_groups_and_positive_mass_overdraw() {
    let s = source(0.5, false);
    let overlap_negative = vec![candidate(&s, 0, false), candidate(&s, 0, false)];
    assert!(!allocated::valid_allocation(
        &s,
        &overlap_negative,
        &mut budget()
    ));
    let mut overdraw = vec![candidate(&s, 0, false), candidate(&s, 1, false)];
    overdraw[1].positive = signed::PositiveTerms(overdraw[0].positive.0.clone());
    assert!(!allocated::valid_allocation(&s, &overdraw, &mut budget()));
    let mut honest = vec![candidate(&s, 0, false), candidate(&s, 1, false)];
    let quarter = (s.q[0].lo * 0.25).next_down();
    for c in &mut honest {
        c.positive = signed::PositiveTerms(vec![(0, quarter)]);
    }
    assert!(allocated::valid_allocation(&s, &honest, &mut budget()));
}

#[test]
fn allocated_outward_mass_budget_refuses_next_up_roundoff_overallocation() {
    let s = source(0.5, false);
    let half = (s.q[0].lo * 0.5).next_up();
    let mut bad = vec![candidate(&s, 0, false), candidate(&s, 1, false)];
    for c in &mut bad {
        c.positive = signed::PositiveTerms(vec![(0, half)]);
    }
    assert!(!allocated::valid_allocation(&s, &bad, &mut budget()));
}

#[test]
fn allocated_overdrawn_duals_and_final_head_residual_cannot_certify_unsafe_source() {
    let mut authored = spec(-0.5, false);
    authored.final_weight[2 * 95] = -1.5;
    let s = Source::build(&authored, &[-1.0; INPUTS], &[1.0; INPUTS], 0, 1, end()).unwrap();
    let cuts = manual_cuts(&s, false, &mut budget());
    let model = triangle::Model::from_root_portfolio(&s, &mut None, &cuts, end()).unwrap();
    assert!(model.request().objective[INPUTS + 95] < -1.49);
    for pair in [[1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [3.0, 3.0]] {
        let mut duals = vec![0.0; model.request().rows.len()];
        let first = duals.len() - 2;
        duals[first..].copy_from_slice(&pair);
        let hint = HeadFunctionalLpHint { row_duals: duals };
        let lower = model.replay(&s, &hint, 1.0, end()).unwrap();
        assert!(lower < -1.0, "{pair:?}: {lower}");
    }
}

#[test]
fn allocated_gray_certificate_refuses_foreign_source_and_mutated_coefficient_bits() {
    let s = source(0.5, false);
    let foreign = source(0.5, false);
    let proposed = candidate(&s, 0, false);
    let cut = signed::certify_gray(
        &s,
        proposed.negative,
        proposed.positive,
        vec![],
        &mut budget(),
    )
    .unwrap();
    assert!(cut.owned_row(&foreign, end()).is_none());
    for eta_mutation in [false, true] {
        let proposed = candidate(&s, 0, false);
        let mut cut = signed::certify_gray(
            &s,
            proposed.negative,
            proposed.positive,
            vec![],
            &mut budget(),
        )
        .unwrap();
        if eta_mutation {
            cut.eta.push((79, 0.5));
        } else {
            cut.direction.heads[0].1 = cut.direction.heads[0].1.next_up();
        }
        assert!(cut.owned_row(&s, end()).is_none());
    }
}

struct ResetFaults;
impl Drop for ResetFaults {
    fn drop(&mut self) {
        EXPIRE_AFTER_MASK.with(|fault| fault.set(None));
        EXPIRE_AFTER_ROUND.with(|fault| fault.set(None));
    }
}

#[test]
fn allocated_shared_mask_cap_is_not_reset_between_independent_rows() {
    let s = source(0.5, false);
    let mut work = Budget {
        masks: MAX_MASKS - 511,
        ..budget()
    };
    let first = candidate(&s, 0, false);
    let first =
        signed::certify_gray(&s, first.negative, first.positive, vec![], &mut work).unwrap();
    let mut prefix = None;
    let model = triangle::Model::from_root_portfolio(&s, &mut prefix, &[first], end()).unwrap();
    let expected = model.request().clone();
    prefix = Some(model.into_phase_prefix());
    let second = candidate(&s, 1, false);
    assert!(
        signed::certify_gray(&s, second.negative, second.positive, vec![], &mut work).is_none()
    );
    assert_eq!(work.masks, MAX_MASKS - 255);
    let (_, retained) = prefix.unwrap().into_phase_parts(&s, 1028, end()).unwrap();
    assert_eq!(retained.len(), expected.rows.len());
    for (row, exported) in retained.iter().zip(&expected.rows) {
        assert_eq!(row.lower.lo.to_bits(), exported.lower.to_bits());
        assert_eq!(
            row.coefficients
                .iter()
                .map(|&(j, a)| (j, 0.5 * a.lo + 0.5 * a.hi))
                .filter(|&(_, a)| a != 0.0)
                .collect::<Vec<_>>(),
            exported.coefficients
        );
    }
}

#[test]
fn allocated_work_cap_and_expired_authority_do_not_start_a_new_certificate() {
    let s = source(0.5, false);
    for expired in [false, true] {
        let proposed = candidate(&s, 0, false);
        let mut work = if expired {
            Budget {
                deadline: Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap(),
                ..budget()
            }
        } else {
            Budget {
                cells: MAX_CELLS - 1,
                ..budget()
            }
        };
        assert!(
            signed::certify_gray(&s, proposed.negative, proposed.positive, vec![], &mut work)
                .is_none()
        );
        assert_eq!(work.masks, 0);
        assert!(work.cells <= MAX_CELLS);
    }
}

#[test]
fn allocated_last_mask_expiry_cannot_install_a_partial_row() {
    let _reset = ResetFaults;
    let s = source(0.5, false);
    let proposed = candidate(&s, 0, false);
    EXPIRE_AFTER_MASK.with(|fault| fault.set(Some(256)));
    let mut work = budget();
    assert!(
        signed::certify_gray(&s, proposed.negative, proposed.positive, vec![], &mut work).is_none()
    );
    assert_eq!(work.masks, 256);
}

#[test]
fn allocated_first_completed_row_survives_expiry_inside_second_gray_cover() {
    let _reset = ResetFaults;
    let s = source(-0.5, false);
    let baseline = s.baseline(end()).unwrap();
    EXPIRE_AFTER_MASK.with(|fault| fault.set(Some(257)));
    let (report, prefix) = refine_with_prefix(&s, baseline, None, end(), end(), &|| None);
    assert_eq!(report.cuts, 1);
    assert_eq!(report.attempted, 2);
    assert_eq!(report.certified, 1);
    assert_eq!(report.masks, 257);
    assert_eq!(report.native_replayed, 1);
    assert_eq!(report.calls, 0);
    assert_eq!(report.ay_replayed, 0);
    assert!(
        report.lower > baseline + 1.0 && report.lower <= -0.5,
        "completed native row lost after later proof expiry: {} -> {}",
        baseline,
        report.lower
    );
    let (_, rows) = prefix
        .expect("first completed row retains publication authority")
        .into_phase_parts(&s, 1028, end())
        .unwrap();
    let mixed = rows
        .iter()
        .filter(|row| {
            row.coefficients
                .iter()
                .filter(|&&(j, a)| j >= INPUTS && a.hi < 0.0)
                .count()
                >= 2
                && row
                    .coefficients
                    .iter()
                    .any(|&(j, a)| j >= INPUTS && a.lo > 0.0)
        })
        .count();
    assert_eq!(mixed, 1);
    check_caps(&report);
}

#[test]
fn allocated_completed_first_batch_survives_expiry_before_the_next_round() {
    let _reset = ResetFaults;
    let s = source(-0.5, false);
    EXPIRE_AFTER_ROUND.with(|fault| fault.set(Some(1)));
    let (report, prefix) =
        refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
    assert!(report.cuts >= 2 && report.native_replayed >= 1);
    assert!(
        report.lower > -0.6 && report.lower <= -0.5,
        "{}",
        report.lower
    );
    assert_eq!(report.lower.to_bits(), report.rounds[0].after.to_bits());
    assert_eq!(report.calls, 0);
    assert!(prefix.is_some());
    check_caps(&report);
}

#[derive(Clone, Copy)]
enum Fault {
    Missing,
    Nan,
    Panic,
    Valid(f64),
    Late,
}

struct MockAdvice {
    fault: Fault,
    calls: AtomicUsize,
}
impl HeadPhaseAdvice for MockAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(request.validate());
        assert_eq!(request.feature_count, INPUTS);
        assert_eq!(request.objective.len(), INPUTS + HEADS);
        assert!(
            mixed_rows(request) >= 2,
            "same original-q LP lost the allocated batch"
        );
        let mut row_duals = request
            .rows
            .iter()
            .map(|row| {
                if row
                    .coefficients
                    .iter()
                    .filter(|&&(j, a)| j >= INPUTS && a < 0.0)
                    .count()
                    >= 2
                    && row
                        .coefficients
                        .iter()
                        .any(|&(j, a)| j >= INPUTS && a > 0.0)
                {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        match self.fault {
            Fault::Missing => return None,
            Fault::Nan => row_duals[0] = f64::NAN,
            Fault::Panic => panic!("mock allocated-row advice fault"),
            Fault::Valid(sign) => row_duals.iter_mut().for_each(|p| *p *= sign),
            Fault::Late => std::thread::sleep(
                deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
            ),
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}

#[test]
fn allocated_missing_nan_and_panicking_hints_preserve_completed_native_batch() {
    for fault in [Fault::Missing, Fault::Nan, Fault::Panic] {
        let s = source(-0.5, false);
        let advice = Arc::new(MockAdvice {
            fault,
            calls: AtomicUsize::new(0),
        });
        let (report, prefix) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| {
                Some(advice.clone())
            });
        assert!(
            report.lower > -0.6 && report.lower <= -0.5,
            "{}",
            report.lower
        );
        assert!(report.cuts >= 2 && report.native_replayed >= 1 && prefix.is_some());
        assert_eq!(report.ay_replayed, 0); // these mocks grant no replay authority
        assert!(report.calls >= 1);
        assert_eq!(report.calls, advice.calls.load(Ordering::Relaxed));
        check_caps(&report);
    }
}

#[test]
fn allocated_valid_both_sign_mock_hints_are_source_replayed_in_the_same_full_q_lp() {
    for sign in [1.0, -1.0] {
        let s = source(-0.5, false);
        let advice = Arc::new(MockAdvice {
            fault: Fault::Valid(sign),
            calls: AtomicUsize::new(0),
        });
        let (report, prefix) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| {
                Some(advice.clone())
            });
        assert!(report.lower > -0.6 && report.lower <= -0.5);
        assert!(report.ay_replayed >= 2 && report.ay_replayed <= 2 * MAX_ROUNDS);
        assert!(report.cuts >= 2 && prefix.is_some());
        assert_eq!(report.calls, advice.calls.load(Ordering::Relaxed));
        check_caps(&report);
    }
}

#[test]
fn allocated_late_hint_cannot_erase_completed_native_floor_or_rows() {
    let s = source(-0.5, false);
    let advice = Arc::new(MockAdvice {
        fault: Fault::Late,
        calls: AtomicUsize::new(0),
    });
    let stage = Instant::now() + Duration::from_secs(2);
    let (report, prefix) =
        refine_with_prefix(&s, s.baseline(end()).unwrap(), None, stage, end(), &|| {
            Some(advice.clone())
        });
    assert_eq!(report.calls, 1);
    assert_eq!(report.ay_replayed, 0);
    assert!(report.lower > -0.6 && report.lower <= -0.5);
    assert!(report.cuts >= 2 && report.native_replayed >= 1 && prefix.is_some());
    assert_eq!(report.lower.to_bits(), report.rounds[0].after.to_bits());
    check_caps(&report);
}

#[test]
fn allocated_default_on_ordinary_root_clock_closes_safe_and_preserves_unsafe_zero() {
    for bias in [0.5, -0.5] {
        let policy = HeadGroupPolicy::new(5.0, 2, 4).unwrap();
        assert!(policy.directional_cuts() && policy.phase_cover().is_none());
        // Ordinary route retains its own750ms root clock; no manual refinement.
        let outcome = verify_with_provider(
            &spec(bias, false),
            &[-1.0; INPUTS],
            &[1.0; INPUTS],
            0,
            1,
            policy,
            end(),
            || None,
        );
        let report = match outcome {
            InputHeadOutcome::Unsat(report, token) if bias > 0.0 => {
                assert_eq!(token.scope(), TokenScope::Instance);
                assert_eq!(token.classes(), &[1]);
                report
            }
            InputHeadOutcome::Unknown(report) if bias < 0.0 => report,
            InputHeadOutcome::Unknown(report) => {
                panic!("safe ordinary allocated route open: {report:?}")
            }
            InputHeadOutcome::Unsat(report, _) => {
                panic!("unsafe allocated route certified: {report:?}")
            }
        };
        let lower = report.lower.unwrap();
        if bias > 0.0 {
            assert!(lower > 0.4 && lower <= 0.5, "{report:?}");
        } else {
            assert!(lower <= -0.5, "{report:?}");
        }
        assert!(report.directional_cuts >= 2 && report.directional_native_replayed >= 1);
        assert!(report.directional_attempted <= 32 && report.directional_rounds <= 3);
        assert!(report.directional_masks <= 8192 && report.directional_work_cells <= 2_000_000);
        assert_eq!(report.advice_calls, 0);
        assert_eq!(report.directional_advice_calls, 0);
        assert_eq!(report.directional_ay_replayed, 0);
    }
}
