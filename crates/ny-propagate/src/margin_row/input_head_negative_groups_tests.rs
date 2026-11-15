// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}
fn expired() -> Instant {
    Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
}
fn rational(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn spec(bias: f32) -> InputHeadSpec {
    InputHeadSpec::new(
        2,
        2,
        vec![1.0, 1.0, 1.0, -1.0],
        vec![0.0; 2],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [71; 32],
    )
    .unwrap()
}
fn source(bias: f32) -> Source {
    Source::build(&spec(bias), &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap()
}
fn group() -> Group {
    Group {
        heads: vec![(0, 1.0), (1, 1.0)],
    }
}
fn certified(s: &Source, g: &Group) -> OwnedRow {
    owned_group_row(s, g, &mut Budget::new(deadline(), MAX_WORK_CELLS)).unwrap()
}
fn append(s: &Source, row: OwnedRow) -> Model<'_> {
    let mut m = Model::build_with_stages(s, false, false, false, deadline()).unwrap();
    assert!(admit_optional_row(
        &m.rows,
        &row,
        m.boxes.len(),
        s.spec.heads
    ));
    m.request.rows.push(HeadFunctionalLpRow {
        coefficients: row
            .coefficients
            .iter()
            .map(|&(j, a)| (j, midpoint(a).unwrap()))
            .filter(|&(_, a)| a != 0.0)
            .collect(),
        lower: row.lower.lo,
    });
    m.rows.push(row);
    assert!(m.request.validate());
    m
}
fn last_hint(m: &Model<'_>, p: f64) -> HeadFunctionalLpHint {
    let mut row_duals = vec![0.0; m.rows.len()];
    *row_duals.last_mut().unwrap() = p;
    HeadFunctionalLpHint { row_duals }
}
fn group_hint(m: &Model<'_>) -> HeadFunctionalLpHint {
    let at = m
        .rows
        .iter()
        .position(|r| {
            r.coefficients.len() == 2 && r.coefficients.iter().all(|&(j, a)| j >= 2 && a.hi < 0.0)
        })
        .unwrap();
    let mut row_duals = vec![0.0; m.rows.len()];
    row_duals[at] = 1.0;
    HeadFunctionalLpHint { row_duals }
}
fn rows_equal(a: &Model<'_>, b: &Model<'_>) {
    assert_eq!(a.rows.len(), b.rows.len());
    for (a, b) in a.rows.iter().zip(&b.rows) {
        assert_eq!(
            (a.lower.lo.to_bits(), a.lower.hi.to_bits()),
            (b.lower.lo.to_bits(), b.lower.hi.to_bits())
        );
        assert_eq!(
            a.coefficients
                .iter()
                .map(|&(j, a)| (j, a.lo.to_bits(), a.hi.to_bits()))
                .collect::<Vec<_>>(),
            b.coefficients
                .iter()
                .map(|&(j, a)| (j, a.lo.to_bits(), a.hi.to_bits()))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn joint_masks_close_fractional_point_admitted_by_all_actual_independent_triangles() {
    let s = source(33.0 / 16.0);
    let old = Model::build_with_stages(&s, true, true, false, deadline()).unwrap();
    assert_eq!(old.cuts(), 0);
    let point = [1.0, 0.0, 1.5, 1.5];
    for (&x, b) in point.iter().zip(&old.boxes) {
        assert!(b.lo <= x && x <= b.hi);
    }
    for row in &old.rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
            sum.add(a.mul(I::p(point[j])).unwrap()).unwrap()
        });
        assert!(
            lhs.lo >= row.lower.hi,
            "the actual complete old row admits the fractional point"
        );
    }
    assert_eq!(rational(33, 16) - rational(3, 1), rational(-15, 16));
    let hint = HeadFunctionalLpHint {
        row_duals: vec![0.25; old.rows.len()],
    };
    assert!(old.replay(&s, &hint, 1.0, deadline()).unwrap() <= -15.0 / 16.0);
    let new = Model::build(&s, deadline()).unwrap();
    assert_eq!(new.cuts(), 0);
    assert_eq!(new.negative_groups(), 1);
    assert_eq!(new.negative_group_masks(), 4);
    assert_eq!(new.rows.len(), old.rows.len() + 1);
    for (a, b) in old.rows.iter().zip(&new.rows) {
        assert_eq!(
            (a.lower.lo.to_bits(), a.lower.hi.to_bits()),
            (b.lower.lo.to_bits(), b.lower.hi.to_bits())
        );
        assert_eq!(
            a.coefficients
                .iter()
                .map(|&(j, a)| (j, a.lo.to_bits(), a.hi.to_bits()))
                .collect::<Vec<_>>(),
            b.coefficients
                .iter()
                .map(|&(j, a)| (j, a.lo.to_bits(), a.hi.to_bits()))
                .collect::<Vec<_>>()
        );
    }
    let lower = new.replay(&s, &group_hint(&new), 1.0, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
}

struct Advice {
    calls: AtomicUsize,
    first: AtomicUsize,
    second: AtomicUsize,
    legacy_variables: usize,
    triangle_variables: usize,
}
impl HeadPhaseAdvice for Advice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        r: &HeadFunctionalLpRequest,
        d: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        assert!(r.validate() && live(d));
        assert_eq!(r.feature_count, 2);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut row_duals = vec![0.0; r.rows.len()];
        if r.bounds.len() == self.triangle_variables {
            self.second.fetch_add(1, Ordering::SeqCst);
            let at = r
                .rows
                .iter()
                .position(|row| {
                    row.coefficients.len() == 2
                        && row
                            .coefficients
                            .iter()
                            .all(|&(j, a)| (2..4).contains(&j) && a < 0.0)
                })
                .unwrap();
            row_duals[at] = 1.0;
        } else {
            assert_eq!(r.bounds.len(), self.legacy_variables);
            assert!(!r.rows.is_empty());
            self.first.fetch_add(1, Ordering::SeqCst);
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}
fn outcome(bias: f32) -> (InputHeadOutcome, Arc<Advice>) {
    run_outcome(bias, true)
}

fn run_outcome(bias: f32, positive_zero_head: bool) -> (InputHeadOutcome, Arc<Advice>) {
    // The third neuron is genuinely authored W=0,b=0,q=+1, so its
    // mathematical ReLU is0 on EVERY original input. Its independent FP32
    // envelope still participates. It admits real legacy epigraph rows;
    // neither a fabricated request row nor an altered class is supplied.
    let authored = if positive_zero_head {
        InputHeadSpec::new(
            2,
            3,
            vec![1.0, 1.0, 0.0, 1.0, -1.0, 0.0],
            vec![0.0; 3],
            vec![-1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            vec![bias, 0.0],
            [79; 32],
        )
        .unwrap()
    } else {
        spec(bias)
    };
    let advice = Arc::new(Advice {
        calls: AtomicUsize::new(0),
        first: AtomicUsize::new(0),
        second: AtomicUsize::new(0),
        legacy_variables: if positive_zero_head { 3 } else { 2 },
        triangle_variables: if positive_zero_head { 5 } else { 4 },
    });
    let result = verify_with_provider(
        &authored,
        &[-1.0; 2],
        &[1.0; 2],
        0,
        1,
        // Test the original joint-group replay independently of later stages.
        HeadGroupPolicy::new(1.0, 1, 1)
            .unwrap()
            .with_directional_cuts(false),
        deadline(),
        || Some(Arc::clone(&advice) as Arc<dyn HeadPhaseAdvice>),
    );
    (result, advice)
}

#[test]
fn complete_joint_row_uses_two_existing_calls_and_original_singleton_instance_authorizer() {
    let (InputHeadOutcome::Unsat(report, token), advice) = outcome(33.0 / 16.0) else {
        panic!("the complete source row must reach the original singleton authorizer")
    };
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.classes(), &[1]);
    assert!(token.covers_all(&[1]) && !token.covers_all(&[0, 1]));
    assert!(report.lower.unwrap() > 1.0 / 32.0 && report.lower.unwrap() <= 1.0 / 16.0);
    assert_eq!(report.advice_calls, 2);
    assert_eq!(report.triangle_advice_calls, 1);
    assert_eq!(report.triangle_variables, 5);
    assert_eq!(report.triangle_cuts, 0);
    assert_eq!(report.triangle_negative_groups, 1);
    assert_eq!(report.triangle_negative_group_masks, 4);
    assert_eq!(report.replayed_candidates, 5);
    assert_eq!(report.triangle_replayed_candidates, 2);
    assert_eq!(advice.calls.load(Ordering::SeqCst), 2);
    assert_eq!(advice.first.load(Ordering::SeqCst), 1);
    assert_eq!(advice.second.load(Ordering::SeqCst), 1);
}

#[test]
fn unsafe_tie_and_negative_stay_open_after_real_joint_row_arrival() {
    for bias in [2.0, 31.0 / 16.0] {
        let (InputHeadOutcome::Unknown(report), advice) = outcome(bias) else {
            panic!("unsafe source cannot publish")
        };
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.triangle_variables, 5);
        assert_eq!(report.replayed_candidates, 5);
        assert_eq!(advice.first.load(Ordering::SeqCst), 1);
        assert_eq!(advice.second.load(Ordering::SeqCst), 1);
        assert!(report.lower.unwrap() <= f64::from(bias) - 2.0 && report.lower.unwrap() <= 0.0);
    }
}

#[test]
fn every_four_head_mask_is_completed_and_arbitrary_duals_bound_exact_dyadic_network() {
    let authored = InputHeadSpec::new(
        2,
        4,
        vec![1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        vec![0.0; 4],
        vec![-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        vec![65.0 / 16.0, 0.0],
        [72; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let g = Group {
        heads: (0..4).map(|j| (j, 1.0)).collect(),
    };
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let row = owned_group_row(&s, &g, &mut budget).unwrap();
    assert_eq!(budget.report.complete_masks, 16);
    assert!(row.lower.lo <= -2.0 && row.lower.lo > -2.001);
    let m = append(&s, row);
    for p in [0.0, 0.25, 1.0, 2.0] {
        let lower = m.replay(&s, &last_hint(&m, p), 1.0, deadline()).unwrap();
        for xn in -4..=4 {
            for zn in -4..=4 {
                let x = rational(xn, 4);
                let z = rational(zn, 4);
                let heads = [&x + &z, &x - &z, -&x + &z, -&x - &z];
                let actual = heads
                    .into_iter()
                    .fold(rational(65, 16), |b, y| b - y.max(rational(0, 1)));
                assert!(BigRational::from_float(lower).unwrap() <= actual);
            }
        }
    }
}

#[test]
fn last_authored_feature_is_present_in_every_selected_mask_support() {
    let authored = InputHeadSpec::new(
        3,
        2,
        vec![1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
        vec![0.0; 2],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![33.0 / 16.0, 0.0],
        [73; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0; 3], &[1.0; 3], 0, 1, deadline()).unwrap();
    let row = certified(&s, &group());
    assert!(row.lower.lo <= -4.0);
    let m = append(&s, row);
    assert!(m.replay(&s, &last_hint(&m, 1.0), 1.0, deadline()).unwrap() <= -31.0 / 16.0);
}

#[test]
fn remaining_head_and_crossing_full_q_debt_are_never_dropped() {
    let authored = InputHeadSpec::new(
        2,
        3,
        vec![1.0, 1.0, 0.0, 1.0, -1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![-1.0, 0.0, -1.0, 0.0, -0.125, 0.0],
        vec![33.0 / 16.0, 0.0],
        [74; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let m = append(&s, certified(&s, &group()));
    assert!(m.replay(&s, &last_hint(&m, 1.0), 1.0, deadline()).unwrap() <= -1.0 / 16.0);
    let mut uncertain = source(33.0 / 16.0);
    uncertain.q[0] = I {
        lo: -1.125,
        hi: 0.125,
    };
    let m = append(&uncertain, certified(&uncertain, &group()));
    assert!(
        m.replay(&uncertain, &last_hint(&m, 1.0), 1.0, deadline())
            .unwrap()
            <= -3.0 / 16.0
    );
}

#[test]
fn independent_first_error_and_final_bias_output_error_weaken_every_owned_row() {
    let mut s = source(33.0 / 16.0);
    s.first_error = vec![0.125; 2];
    let row = certified(&s, &group());
    assert!(row.lower.lo <= -2.25);
    let m = append(&s, row);
    assert!(m.replay(&s, &last_hint(&m, 1.0), 1.0, deadline()).unwrap() <= -3.0 / 16.0);
    let mut final_debt = source(33.0 / 16.0);
    final_debt.bias = I {
        lo: 31.0 / 16.0,
        hi: 35.0 / 16.0,
    };
    final_debt.output_error = 0.125;
    let m = append(&final_debt, certified(&final_debt, &group()));
    assert!(
        m.replay(&final_debt, &last_hint(&m, 1.0), 1.0, deadline())
            .unwrap()
            <= -3.0 / 16.0
    );
}

#[test]
fn unequal_and_overdrawn_masses_keep_full_residuals_and_exact_source_bounds() {
    let s = source(33.0 / 16.0);
    for weights in [[0.25, 2.0], [2.0, 0.5], [4.0, 4.0]] {
        let g = Group {
            heads: vec![(0, weights[0]), (1, weights[1])],
        };
        let m = append(&s, certified(&s, &g));
        for p in [0.5, 1.0, 2.0] {
            let lower = m.replay(&s, &last_hint(&m, p), 1.0, deadline()).unwrap();
            assert!(lower <= 1.0 / 16.0);
            for x in [-1.0_f64, 0.0, 1.0] {
                for z in [-1.0_f64, 0.0, 1.0] {
                    assert!(lower <= 33.0 / 16.0 - (x + z).max(0.0) - (x - z).max(0.0));
                }
            }
        }
    }
}

#[test]
fn half_eta_first_error_is_charged_before_large_later_dual_amplification() {
    let authored = InputHeadSpec::new(
        1,
        2,
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 4],
        vec![0.0; 2],
        [75; 32],
    )
    .unwrap();
    let mut s = Source::build(&authored, &[0.0], &[0.0], 0, 1, deadline()).unwrap();
    let eta = f64::from_bits(1);
    s.first_error = vec![eta; 2];
    s.y = vec![I::p(0.0); 2];
    s.q = vec![I::p(0.0); 2];
    s.bias = I::p(0.0);
    s.output_error = 0.0;
    let g = Group {
        heads: vec![(0, 0.5), (1, 0.5)],
    };
    let row = certified(&s, &g);
    assert!(row.lower.lo <= -eta);
    let m = append(&s, row);
    let large = 2.0f64.powi(256);
    assert!(
        m.replay(&s, &last_hint(&m, large), 1.0, deadline())
            .unwrap()
            <= -large * eta
    );
}

#[test]
fn expiry_after_real_mask_support_never_installs_partial_max_row() {
    let s = source(33.0 / 16.0);
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(1)));
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    assert!(owned_group_row(&s, &group(), &mut budget).is_none());
    assert_eq!(budget.report.complete_masks, 1);
    assert!(budget.report.expired);
    EXPIRE_AFTER_MASK.with(|f| f.set(None));
    let old = Model::build_with_stages(&s, true, true, false, deadline()).unwrap();
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(1)));
    let declined = Model::build(&s, deadline()).unwrap();
    EXPIRE_AFTER_MASK.with(|f| f.set(None));
    assert_eq!(declined.negative_groups(), 0);
    rows_equal(&old, &declined);
}

#[test]
fn expiry_during_later_group_retains_only_prior_complete_row() {
    let authored = InputHeadSpec::new(
        2,
        3,
        vec![1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        vec![0.0; 3],
        vec![-1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        vec![4.0, 0.0],
        [76; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(5)));
    let (completed, report) = rows(&s, deadline());
    EXPIRE_AFTER_MASK.with(|f| f.set(None));
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].1, 4);
    assert_eq!(report.checked, 1);
    assert_eq!(report.complete_masks, 5);
    assert!(report.expired);
    assert!(completed[0].0.valid(5));
    let old = Model::build_with_stages(&s, true, true, false, deadline()).unwrap();
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(5)));
    let kept = Model::build(&s, deadline()).unwrap();
    EXPIRE_AFTER_MASK.with(|f| f.set(None));
    assert_eq!(kept.negative_groups(), 1);
    assert_eq!(kept.negative_group_masks(), 4);
    assert_eq!(kept.rows.len(), old.rows.len() + 1);
}

#[test]
fn expiry_after_actual_positive_replay_rolls_back_to_original_completed_floor() {
    let s = source(33.0 / 16.0);
    let before = s.baseline(deadline()).unwrap();
    let m = Model::build(&s, deadline()).unwrap();
    let improved = m.replay(&s, &group_hint(&m), 1.0, deadline()).unwrap();
    assert!(before < 0.0 && improved > 1.0 / 32.0);
    assert_eq!(
        commit_triangle_best(before, improved, deadline(), expired()).to_bits(),
        before.to_bits()
    );
    assert_eq!(
        commit_triangle_best(before, improved, expired(), deadline()).to_bits(),
        before.to_bits()
    );
    assert_eq!(
        commit_triangle_best(before, improved, deadline(), deadline()).to_bits(),
        improved.to_bits()
    );
}

#[test]
fn equal_bytes_foreign_generation_has_no_joint_replay_authority() {
    let s = source(33.0 / 16.0);
    let foreign = source(33.0 / 16.0);
    let m = Model::build(&s, deadline()).unwrap();
    let other = Model::build(&foreign, deadline()).unwrap();
    assert_eq!(m.request.bounds, other.request.bounds);
    assert_eq!(m.request.objective, other.request.objective);
    assert!(!Arc::ptr_eq(&s.spec, &foreign.spec));
    assert!(m
        .replay(&foreign, &group_hint(&m), 1.0, deadline())
        .is_none());
}

#[test]
fn duplicate_unsorted_negative_nonfinite_and_oversized_groups_refuse_before_any_mask() {
    let s = source(33.0 / 16.0);
    for heads in [
        vec![(0, 1.0)],
        vec![(0, 1.0), (0, 1.0)],
        vec![(1, 1.0), (0, 1.0)],
        vec![(0, -1.0), (1, 1.0)],
        vec![(0, f64::NAN), (1, 1.0)],
        vec![(0, f64::INFINITY), (1, 1.0)],
        vec![(0, 1.0), (2, 1.0)],
        vec![(0, 1.0); 5],
    ] {
        let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
        assert!(owned_group_row(&s, &Group { heads }, &mut budget).is_none());
        assert_eq!(budget.report.complete_masks, 0);
    }
}

#[test]
fn total_256_masks_cap_is_shared_across_complete_groups_without_partial_authority() {
    let authored = InputHeadSpec::new(
        1,
        4,
        vec![1.0, -1.0, 0.5, -0.5],
        vec![0.0; 4],
        vec![-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        vec![4.0, 0.0],
        [77; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
    let g = Group {
        heads: (0..4).map(|j| (j, 1.0)).collect(),
    };
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    for _ in 0..16 {
        assert!(owned_group_row(&s, &g, &mut budget).is_some());
    }
    assert_eq!(budget.report.complete_masks, 256);
    assert!(owned_group_row(&s, &g, &mut budget).is_none());
    assert_eq!(budget.report.complete_masks, 256);
}

#[test]
fn work_and_rank_caps_refuse_without_manufactured_complete_support() {
    let s = source(33.0 / 16.0);
    let mut budget = Budget::new(deadline(), 0);
    assert!(owned_group_row(&s, &group(), &mut budget).is_none());
    assert!(budget.report.work_exhausted);
    assert_eq!(budget.report.complete_masks, 0);
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    budget.report.rank_trials = MAX_RANK_TRIALS;
    assert!(rank(&s, &group(), &mut budget).is_none());
    assert_eq!(budget.report.rank_trials, 512);
    assert_eq!(budget.report.complete_masks, 0);
}

#[test]
fn optional_wire_or_magnitude_refusal_preserves_every_prior_model_row() {
    let s = source(33.0 / 16.0);
    let old = Model::build_with_stages(&s, true, true, false, deadline()).unwrap();
    let row = OwnedRow {
        coefficients: vec![(2, I::p(-1.0))],
        lower: I::p(-1e101),
    };
    assert!(!admit_optional_row(&old.rows, &row, 4, 2));
    let again = Model::build_with_stages(&s, true, true, false, deadline()).unwrap();
    rows_equal(&old, &again);
    let row = certified(&s, &group());
    assert!(!admit_optional_row(&old.rows, &row, usize::MAX, 2));
}

#[test]
fn no_two_unstable_negative_heads_leaves_legacy_geometry_unchanged() {
    let authored = InputHeadSpec::new(
        2,
        2,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.0, 2.0],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![4.0, 0.0],
        [78; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    assert!(!eligible(&s));
    let (rows, report) = rows(&s, deadline());
    assert!(rows.is_empty());
    assert_eq!(report.rank_trials, 0);
    assert_eq!(report.complete_masks, 0);
    let old = Model::build_with_stages(&s, true, true, false, deadline()).unwrap();
    let new = Model::build(&s, deadline()).unwrap();
    rows_equal(&old, &new);
    assert_eq!(new.negative_groups(), 0);
}

#[test]
fn all_negative_legacy_decline_still_reaches_original_instance_with_one_real_call() {
    let s = source(33.0 / 16.0);
    assert!(
        s.export(deadline()).is_none(),
        "no positive epigraph rows means no valid legacy request"
    );
    let (InputHeadOutcome::Unsat(report, token), advice) = run_outcome(33.0 / 16.0, false) else {
        panic!("one genuine triangle callback must still reach the original singleton")
    };
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.classes(), &[1]);
    assert!(token.covers_all(&[1]) && !token.covers_all(&[0, 1]));
    assert!(report.lower.unwrap() > 1.0 / 32.0 && report.lower.unwrap() <= 1.0 / 16.0);
    assert_eq!(report.advice_calls, 1);
    assert_eq!(report.triangle_advice_calls, 1);
    assert_eq!(report.triangle_variables, 4);
    assert_eq!(report.triangle_cuts, 0);
    assert_eq!(report.triangle_negative_groups, 1);
    assert_eq!(report.triangle_negative_group_masks, 4);
    assert_eq!(report.replayed_candidates, 3);
    assert_eq!(report.triangle_replayed_candidates, 2);
    assert_eq!(advice.calls.load(Ordering::SeqCst), 1);
    assert_eq!(advice.first.load(Ordering::SeqCst), 0);
    assert_eq!(advice.second.load(Ordering::SeqCst), 1);
}

#[test]
fn all_negative_tie_and_negative_keep_actual_one_call_arrival_without_unsat() {
    for bias in [2.0, 31.0 / 16.0] {
        let (InputHeadOutcome::Unknown(report), advice) = run_outcome(bias, false) else {
            panic!("all-negative unsafe input must never acquire an instance token")
        };
        assert_eq!(report.advice_calls, 1);
        assert_eq!(report.triangle_advice_calls, 1);
        assert_eq!(report.triangle_variables, 4);
        assert_eq!(report.triangle_negative_groups, 1);
        assert_eq!(report.triangle_negative_group_masks, 4);
        assert_eq!(report.replayed_candidates, 3);
        assert_eq!(report.triangle_replayed_candidates, 2);
        assert_eq!(advice.calls.load(Ordering::SeqCst), 1);
        assert_eq!(advice.first.load(Ordering::SeqCst), 0);
        assert_eq!(advice.second.load(Ordering::SeqCst), 1);
        assert!(report.lower.unwrap() <= f64::from(bias) - 2.0 && report.lower.unwrap() <= 0.0);
    }
}
