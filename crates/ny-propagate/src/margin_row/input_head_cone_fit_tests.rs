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

fn four_spec(bias: f32) -> InputHeadSpec {
    let mut first = vec![0.0; 4 * 5];
    for k in 0..4 {
        first[k * 5 + k] = 1.0;
        first[k * 5 + 4] = 0.25;
    }
    let mut final_weight = vec![0.0; 5 * 2];
    for j in 0..4 {
        final_weight[j * 2] = 0.25;
    }
    final_weight[8] = -1.0;
    InputHeadSpec::new(
        4,
        5,
        first,
        vec![0.0; 5],
        final_weight,
        vec![bias, 0.0],
        [41; 32],
    )
    .unwrap()
}

fn source(bias: f32) -> Source {
    Source::build(&four_spec(bias), &[-1.0; 4], &[1.0; 4], 0, 1, deadline()).unwrap()
}

fn four_cut() -> MassCut {
    MassCut {
        positive: (0..4).map(|j| (j, 0.25)).collect(),
        negative: 4,
        tau: 1.0,
    }
}

fn cone_hint(model: &Model<'_>) -> HeadFunctionalLpHint {
    let row = model
        .rows
        .iter()
        .enumerate()
        .find(|(_, row)| {
            row.coefficients.len() == 5
                && row
                    .coefficients
                    .iter()
                    .filter(|&&(j, a)| j >= 4 && a.lo > 0.0)
                    .count()
                    == 4
                && row.coefficients.iter().any(|&(j, a)| j == 8 && a.hi < 0.0)
        })
        .expect("the completed source-owned four-mass row must exist")
        .0;
    let mut row_duals = vec![0.0; model.rows.len()];
    row_duals[row] = 1.0;
    HeadFunctionalLpHint { row_duals }
}

fn manual_model<'a>(source: &'a Source, cut: &MassCut) -> Model<'a> {
    let mut model = Model::build_with_options(source, false, false, deadline()).unwrap();
    let row = owned_cut_row(source, cut, deadline()).unwrap();
    model.request.rows.push(HeadFunctionalLpRow {
        coefficients: row
            .coefficients
            .iter()
            .map(|&(j, a)| (j, midpoint(a).unwrap()))
            .filter(|&(_, a)| a != 0.0)
            .collect(),
        lower: row.lower.lo,
    });
    model.rows.push(row);
    model.cuts += 1;
    assert!(model.request.validate());
    model
}

fn last_hint(model: &Model<'_>) -> HeadFunctionalLpHint {
    let mut row_duals = vec![0.0; model.rows.len()];
    *row_duals.last_mut().unwrap() = 1.0;
    HeadFunctionalLpHint { row_duals }
}

fn same_model(a: &Model<'_>, b: &Model<'_>) {
    assert_eq!(a.request.bounds, b.request.bounds);
    assert_eq!(a.request.objective, b.request.objective);
    assert_eq!(a.request.slope_rows, b.request.slope_rows);
    assert_eq!(a.request.phase_rows, b.request.phase_rows);
    assert_eq!(a.rows.len(), b.rows.len());
    assert_eq!(a.cuts, b.cuts);
    for (a, b) in a.rows.iter().zip(&b.rows) {
        assert_eq!(
            (a.lower.lo.to_bits(), a.lower.hi.to_bits()),
            (b.lower.lo.to_bits(), b.lower.hi.to_bits())
        );
        assert_eq!(a.coefficients.len(), b.coefficients.len());
        for (&(ja, a), &(jb, b)) in a.coefficients.iter().zip(&b.coefficients) {
            assert_eq!(ja, jb);
            assert_eq!(
                (a.lo.to_bits(), a.hi.to_bits()),
                (b.lo.to_bits(), b.hi.to_bits())
            );
        }
    }
}

#[test]
fn four_mass_cut_closes_where_every_three_support_representation_is_open() {
    let s = source(1.0 / 16.0);
    let old = Model::build_with_options(&s, true, false, deadline()).unwrap();
    let point = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25];
    for (&value, box_) in point.iter().zip(&old.boxes) {
        assert!(box_.lo <= value && value <= box_.hi);
    }
    for row in &old.rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
            sum.add(a.mul(I::p(point[j])).unwrap()).unwrap()
        });
        assert!(
            lhs.lo >= row.lower.hi,
            "actual old dyadic/triangle rows stay open"
        );
    }
    // Universal algebra in the method note: any <=3-supported nonnegative
    // mass cut misses at least one feature, so its exact source lower is
    // <=-tau/4. Thus r_negative=1/4 satisfies EVERY such cut, not just these
    // enumerated controls. Check exact dyadic overdraw/tau controls too.
    for mask in 0u32..16 {
        if mask.count_ones() > 3 {
            continue;
        }
        for a in [
            rational(1, 8),
            rational(1, 4),
            rational(1, 2),
            rational(2, 1),
        ] {
            for tau in [rational(1, 2), rational(1, 1), rational(2, 1)] {
                let quarter = tau / rational(4, 1);
                let mut absolute = rational(0, 1);
                for j in 0..4 {
                    let coefficient = if mask & (1 << j) != 0 {
                        a.clone() - &quarter
                    } else {
                        -quarter.clone()
                    };
                    absolute += if coefficient < rational(0, 1) {
                        -coefficient
                    } else {
                        coefficient
                    };
                }
                assert!(-absolute <= -quarter);
            }
        }
    }
    assert_eq!(rational(1, 16) - rational(1, 4), rational(-3, 16));
    let arbitrary = HeadFunctionalLpHint {
        row_duals: vec![0.125; old.rows.len()],
    };
    assert!(old.replay(&s, &arbitrary, 1.0, deadline()).unwrap() <= -3.0 / 16.0);
    let new = Model::build(&s, deadline()).unwrap();
    assert_eq!(new.cone_cuts(), 1);
    assert!(new.rows.len() > old.rows.len());
    for (a, b) in old.rows.iter().zip(&new.rows) {
        assert_eq!(a.lower.lo.to_bits(), b.lower.lo.to_bits());
        assert_eq!(a.lower.hi.to_bits(), b.lower.hi.to_bits());
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
    let lower = new.replay(&s, &cone_hint(&new), 1.0, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
}

struct Advice {
    legacy: AtomicUsize,
    triangle: AtomicUsize,
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
        assert_eq!(r.feature_count, 4);
        let mut row_duals = vec![0.0; r.rows.len()];
        if r.bounds.len() == 8 {
            self.legacy.fetch_add(1, Ordering::SeqCst);
        } else {
            assert_eq!(r.bounds.len(), 9);
            self.triangle.fetch_add(1, Ordering::SeqCst);
            let at = r
                .rows
                .iter()
                .position(|row| {
                    row.coefficients.len() == 5
                        && row
                            .coefficients
                            .iter()
                            .filter(|&&(j, a)| j >= 4 && a > 0.0)
                            .count()
                            == 4
                        && row.coefficients.iter().any(|&(j, a)| j == 8 && a < 0.0)
                })
                .unwrap();
            row_duals[at] = 1.0;
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}

#[test]
fn cone_row_reaches_original_singleton_instance_with_exactly_two_existing_calls() {
    let advice = Arc::new(Advice {
        legacy: AtomicUsize::new(0),
        triangle: AtomicUsize::new(0),
    });
    let InputHeadOutcome::Unsat(report, token) = verify_with_provider(
        &four_spec(1.0 / 16.0),
        &[-1.0; 4],
        &[1.0; 4],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        deadline(),
        || Some(Arc::clone(&advice) as Arc<dyn HeadPhaseAdvice>),
    ) else {
        panic!("four-mass full-source replay failed to close the original margin")
    };
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.classes(), &[1]);
    assert!(token.covers_all(&[1]) && !token.covers_all(&[0, 1]));
    assert!(report.lower.unwrap() > 1.0 / 32.0 && report.lower.unwrap() <= 1.0 / 16.0);
    assert_eq!(report.advice_calls, 2);
    assert_eq!(report.triangle_advice_calls, 1);
    assert_eq!(report.replayed_candidates, 5);
    assert_eq!(report.triangle_replayed_candidates, 2);
    assert_eq!(advice.legacy.load(Ordering::SeqCst), 1);
    assert_eq!(advice.triangle.load(Ordering::SeqCst), 1);
}

#[test]
fn full_thirty_feature_fit_reads_last_coordinate_and_bias() {
    let mut first = vec![0.0; 30 * 5];
    let hadamard = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    for k in 0..30 {
        let row = if k < 28 {
            hadamard[k % 4]
        } else if k == 29 {
            hadamard[2]
        } else {
            [0.0; 4]
        };
        for j in 0..4 {
            first[k * 5 + j] = row[j];
        }
        first[k * 5 + 4] = row.iter().sum::<f32>() / 4.0;
    }
    let mut spec = four_spec(1.0 / 16.0);
    spec.inputs = 30;
    spec.first = first;
    let s = Source::build(&spec, &[-1.0; 30], &[1.0; 30], 0, 1, deadline()).unwrap();
    let old = super::super::proposals(&s, deadline()).unwrap();
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let v = vector(&s, 0, &mut budget).unwrap();
    assert_eq!(v.len(), 31);
    assert_eq!(v[29], 1.0);
    let cut = fit(&s, 4, &old, &mut budget).unwrap();
    assert_eq!(cut.positive.len(), 4);
    assert!(budget.report.work_cells <= MAX_WORK_CELLS);
    let model = manual_model(&s, &cut);
    let lower = model
        .replay(&s, &last_hint(&model), 1.0, deadline())
        .unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
    let mut bias_spec = four_spec(1.0 / 16.0);
    bias_spec.first_bias[0] = 0.5;
    let s2 = Source::build(&bias_spec, &[-1.0; 4], &[1.0; 4], 0, 1, deadline()).unwrap();
    assert_eq!(
        vector(&s2, 0, &mut Budget::new(deadline(), MAX_WORK_CELLS)).unwrap()[4],
        0.5
    );
}

#[test]
fn omitted_last_input_and_remaining_head_debt_prevent_nominal_closure() {
    let mut spec = four_spec(1.0 / 16.0);
    spec.inputs = 5;
    spec.first.extend([0.0, 0.0, 0.0, 0.0, 1.0]);
    let s = Source::build(&spec, &[-1.0; 5], &[1.0; 5], 0, 1, deadline()).unwrap();
    let model = manual_model(&s, &four_cut());
    assert!(source_cut_lower(&s, &four_cut(), deadline()).unwrap() <= -1.0);
    assert!(
        model
            .replay(&s, &last_hint(&model), 1.0, deadline())
            .unwrap()
            <= -15.0 / 16.0
    );
    let old = source(1.0 / 16.0);
    let expanded = InputHeadSpec::new(
        4,
        6,
        (0..4)
            .flat_map(|k| {
                (0..6).map(move |j| {
                    if j == k {
                        1.0
                    } else if j == 4 {
                        0.25
                    } else {
                        0.0
                    }
                })
            })
            .collect(),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        vec![
            0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, -1.0, 0.0, -1.0, 0.0,
        ],
        vec![1.0 / 16.0, 0.0],
        [42; 32],
    )
    .unwrap();
    assert_eq!(expanded.heads, old.spec.heads + 1);
    let s = Source::build(&expanded, &[-1.0; 4], &[1.0; 4], 0, 1, deadline()).unwrap();
    let model = manual_model(&s, &four_cut());
    assert!(
        model
            .replay(&s, &last_hint(&model), 1.0, deadline())
            .unwrap()
            <= 1.0 / 16.0 - 2.0
    );
}

#[test]
fn independent_first_error_full_q_bias_and_output_error_survive_four_mass_cancellation() {
    for fault in 0..4 {
        let mut s = source(1.0 / 16.0);
        match fault {
            0 => {
                s.first_error.fill(0.125);
                for y in &mut s.y {
                    y.lo -= 0.125;
                    y.hi += 0.125;
                }
            }
            1 => {
                for j in 0..4 {
                    s.q[j] = I {
                        lo: 0.125,
                        hi: 0.375,
                    };
                }
                s.q[4] = I {
                    lo: -1.25,
                    hi: -0.75,
                };
            }
            2 => {
                s.bias = I {
                    lo: -1.0 / 16.0,
                    hi: 1.0 / 16.0,
                }
            }
            _ => s.output_error = 0.25,
        }
        let model = manual_model(&s, &four_cut());
        assert!(
            model
                .replay(&s, &last_hint(&model), 1.0, deadline())
                .unwrap()
                <= 0.0
        );
    }
}

#[test]
fn direct_overdrawn_nonunit_cone_masses_bound_every_original_control() {
    let s = source(1.0 / 16.0);
    let cut = MassCut {
        positive: vec![(0, 2.0), (1, 0.125), (2, 0.5), (3, 0.25)],
        negative: 4,
        tau: 3.0,
    };
    let lower = source_cut_lower(&s, &cut, deadline()).unwrap();
    for mask in 0..81usize {
        let mut m = mask;
        let mut x = [0.0_f64; 4];
        for v in &mut x {
            *v = (m % 3) as f64 - 1.0;
            m /= 3;
        }
        let actual = cut
            .positive
            .iter()
            .map(|&(j, a)| a * x[j].max(0.0))
            .sum::<f64>()
            - cut.tau * (x.iter().sum::<f64>() / 4.0).max(0.0);
        assert!(lower <= actual);
    }
    let model = manual_model(&s, &cut);
    assert!(
        model
            .replay(&s, &last_hint(&model), 1.0, deadline())
            .unwrap()
            <= 1.0 / 16.0
    );
    let mut duplicate = cut.clone();
    duplicate.positive.push((0, 1.0));
    assert!(source_cut_lower(&s, &duplicate, deadline()).is_none());
    let mut invalid = cut;
    invalid.positive[3].1 = f64::INFINITY;
    assert!(owned_cut_row(&s, &invalid, deadline()).is_none());
}

#[test]
fn four_subnormal_mass_products_are_priced_before_original_input_amplification() {
    let mut first = vec![0.0; 5];
    first[..4].fill(f32::from_bits(1));
    let spec = InputHeadSpec::new(
        1,
        5,
        first,
        vec![0.0; 5],
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        vec![0.0; 2],
        [43; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1e30], &[1e30], 0, 1, deadline()).unwrap();
    let eta = f64::from_bits(1);
    assert_eq!(eta * f64::from(f32::from_bits(1)), 0.0);
    let cut = MassCut {
        positive: (0..4).map(|j| (j, eta)).collect(),
        negative: 4,
        tau: 0.0,
    };
    let lower = source_cut_lower(&s, &cut, deadline()).unwrap();
    assert!(lower.is_finite() && lower < -1e-300);
}

#[test]
fn zero_and_negative_original_margins_never_publish_a_cone_token() {
    for bias in [0.0, -1.0 / 16.0] {
        let advice = Arc::new(Advice {
            legacy: AtomicUsize::new(0),
            triangle: AtomicUsize::new(0),
        });
        let InputHeadOutcome::Unknown(report) = verify_with_provider(
            &four_spec(bias),
            &[-1.0; 4],
            &[1.0; 4],
            0,
            1,
            // Preserve this cone-only tie/negative callback contract.
            HeadGroupPolicy::new(1.0, 1, 1)
                .unwrap()
                .with_directional_cuts(false),
            deadline(),
            || Some(Arc::clone(&advice) as Arc<dyn HeadPhaseAdvice>),
        ) else {
            panic!("unsafe cone margin published")
        };
        assert!(report.lower.unwrap() <= f64::from(bias));
        assert_eq!(report.advice_calls, 2);
        assert_eq!(advice.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn thirty_two_mass_support_is_admitted_and_thirty_three_is_refused() {
    let d = 33;
    let h = 34;
    let mut first = vec![0.0; d * h];
    for k in 0..d {
        first[k * h + k] = 1.0;
        first[k * h + 33] = 1.0 / 32.0;
    }
    let mut final_weight = vec![0.0; h * 2];
    for j in 0..d {
        final_weight[j * 2] = 1.0 / 32.0;
    }
    final_weight[66] = -1.0;
    let spec = InputHeadSpec::new(
        d,
        h,
        first,
        vec![0.0; h],
        final_weight,
        vec![1.0 / 16.0, 0.0],
        [44; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &vec![-1.0; d], &vec![1.0; d], 0, 1, deadline()).unwrap();
    let mut cut = MassCut {
        positive: (0..32).map(|j| (j, 1.0 / 32.0)).collect(),
        negative: 33,
        tau: 1.0,
    };
    let row = owned_cut_row(&s, &cut, deadline()).unwrap();
    assert!(row.valid(d + h));
    assert_eq!(row.coefficients.len(), 33);
    cut.positive.push((32, 1.0 / 32.0));
    assert!(owned_cut_row(&s, &cut, deadline()).is_none());
}

#[test]
fn equal_geometry_fresh_generation_and_original_expiry_have_no_cone_authority() {
    let s = source(1.0 / 16.0);
    let foreign = source(1.0 / 16.0);
    let model = Model::build(&s, deadline()).unwrap();
    let hint = cone_hint(&model);
    let lower = model.replay(&s, &hint, 1.0, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0);
    assert!(!Arc::ptr_eq(&s.spec, &foreign.spec));
    assert!(model.replay(&foreign, &hint, 1.0, deadline()).is_none());
    assert!(model.replay(&s, &hint, 1.0, expired()).is_none());
    assert!(Model::build(&s, expired()).is_none());
    let before = s.baseline(deadline()).unwrap();
    assert!(before < 0.0);
    assert_eq!(
        commit_triangle_best(before, lower, deadline(), expired()).to_bits(),
        before.to_bits()
    );
}

struct ExpiredPhase;
impl Drop for ExpiredPhase {
    fn drop(&mut self) {
        FORCE_EXPIRED_PHASE.with(|v| v.set(false));
    }
}

#[test]
fn optional_fit_phase_expiry_preserves_bit_exact_complete_old_model() {
    let s = source(1.0 / 16.0);
    let old = Model::build_with_options(&s, true, false, deadline()).unwrap();
    assert!(old.cuts() > 0);
    let _reset = ExpiredPhase;
    FORCE_EXPIRED_PHASE.with(|v| v.set(true));
    let new = Model::build(&s, deadline()).unwrap();
    assert_eq!(new.cone_cuts(), 0);
    same_model(&old, &new);
    let hint = HeadFunctionalLpHint {
        row_duals: vec![0.25; old.rows.len()],
    };
    assert_eq!(
        old.replay(&s, &hint, 1.0, deadline()).unwrap().to_bits(),
        new.replay(&s, &hint, 1.0, deadline()).unwrap().to_bits()
    );
}

#[test]
fn fit_work_target_iteration_and_support_caps_are_independent_of_clock() {
    let s = source(1.0 / 16.0);
    let legacy = super::super::proposals(&s, deadline()).unwrap();
    let mut short = Budget::new(deadline(), 8);
    assert!(propose_with_budget(&s, &legacy, &mut short).is_empty());
    assert!(short.report.work_exhausted && !short.report.expired);
    assert!(short.report.work_cells <= 8);
    let mut live_budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let result = propose_with_budget(&s, &legacy, &mut live_budget);
    assert_eq!(result.len(), 1);
    assert_eq!(live_budget.report.support_max, 4);
    assert!(live_budget.report.targets <= MAX_TARGETS);
    assert!(live_budget.report.sweeps <= MAX_TARGETS * MAX_SWEEPS);
    assert!(live_budget.report.coordinates <= MAX_TARGETS * MAX_SWEEPS * MAX_COLUMNS);
    assert!(live_budget.report.work_cells <= MAX_WORK_CELLS);
    assert!(phase_deadline(Instant::now(), expired()) <= Instant::now());
}

#[test]
fn crossing_zero_output_mass_stays_in_complete_residual_even_when_fit_skips_it() {
    let mut s = source(1.0 / 16.0);
    s.q[0] = I {
        lo: -0.25,
        hi: 0.25,
    };
    let model = manual_model(&s, &four_cut());
    // At all original x_i=1, the admissible lower endpoint q_0=-1/4
    // yields margin B-1/2. Positive-cone eligibility cannot omit this head.
    assert!(
        model
            .replay(&s, &last_hint(&model), 1.0, deadline())
            .unwrap()
            <= 1.0 / 16.0 - 0.5
    );
    let legacy = super::super::proposals(&s, deadline()).unwrap();
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let proposals = propose_with_budget(&s, &legacy, &mut budget);
    assert!(!proposals.is_empty());
    for cut in &proposals {
        assert!(cut.positive.iter().all(|&(head, _)| head != 0));
        let offered = manual_model(&s, cut);
        assert!(
            offered
                .replay(&s, &last_hint(&offered), 1.0, deadline())
                .unwrap()
                <= 1.0 / 16.0 - 0.5
        );
    }
    assert!(!budget.report.expired && !budget.report.work_exhausted);
}

#[test]
fn one_mass_duplicate_affine_cut_closes_actual_independent_triangle_gap() {
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [81; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
    let old = Model::build_with_options(&s, true, false, deadline()).unwrap();
    assert_eq!(old.cuts(), 0);
    // A genuine fractional point of the complete old model: both affine
    // heads equal x, but independent triangles permit unequal ReLU values.
    let point = [0.0, 0.0, 0.25];
    for (&v, b) in point.iter().zip(&old.boxes) {
        assert!(b.lo <= v && v <= b.hi);
    }
    for row in &old.rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
            sum.add(a.mul(I::p(point[j])).unwrap()).unwrap()
        });
        assert!(lhs.lo >= row.lower.hi);
    }
    assert_eq!(rational(1, 16) - rational(1, 4), rational(-3, 16));
    let arbitrary = HeadFunctionalLpHint {
        row_duals: vec![0.125; old.rows.len()],
    };
    assert!(old.replay(&s, &arbitrary, 1.0, deadline()).unwrap() <= -3.0 / 16.0);
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let cut = fit(&s, 1, &[], &mut budget).unwrap();
    assert_eq!(cut.positive, vec![(0, 1.0)]);
    let new = Model::build(&s, deadline()).unwrap();
    assert_eq!(new.cone_cuts(), 1);
    let lower = new.replay(&s, &last_hint(&new), 1.0, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
}

#[test]
fn three_stored_float32_masses_close_point_admitted_by_every_two_support_cut() {
    // Use the EXACT stored binary32 mass as a direct represented f64 value;
    // no claimed identity with exact 1/3 or rounded normalization is needed.
    let t32 = 1.0f32 / 3.0;
    let t = f64::from(t32);
    let mut first = vec![0.0; 3 * 4];
    for k in 0..3 {
        first[k * 4 + k] = 1.0;
        first[k * 4 + 3] = t32;
    }
    let mut last = vec![0.0; 4 * 2];
    for j in 0..3 {
        last[j * 2] = t32;
    }
    last[6] = -1.0;
    let spec = InputHeadSpec::new(
        3,
        4,
        first,
        vec![0.0; 4],
        last,
        vec![1.0 / 16.0, 0.0],
        [82; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 3], &[1.0; 3], 0, 1, deadline()).unwrap();
    let old = Model::build_with_options(&s, true, false, deadline()).unwrap();
    let point = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t];
    for (&v, b) in point.iter().zip(&old.boxes) {
        assert!(b.lo <= v && v <= b.hi);
    }
    for row in &old.rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
            sum.add(a.mul(I::p(point[j])).unwrap()).unwrap()
        });
        assert!(lhs.lo >= row.lower.hi);
    }
    // For ANY nonnegative masses on at most two distinct positive heads,
    // one missing coordinate contributes |0-tau*t|. Consequently the exact
    // source floor is -sum|a_i-tau*t| <= -tau*t, and r_n=t at x=0 obeys
    // EVERY such cut. This missing-coordinate argument, rather than the
    // finite controls below, proves the representation gap.
    assert_eq!(t32.to_bits(), 0x3eaa_aaab);
    let t_exact = rational(11_184_811, 33_554_432);
    for mask in 0u32..8 {
        if mask.count_ones() > 2 {
            continue;
        }
        for a in [
            rational(1, 8),
            t_exact.clone(),
            rational(1, 2),
            rational(2, 1),
        ] {
            for tau in [rational(1, 2), rational(1, 1), rational(2, 1)] {
                let target = &tau * &t_exact;
                let mut absolute = rational(0, 1);
                for j in 0..3 {
                    let coefficient = if mask & (1 << j) != 0 {
                        &a - &target
                    } else {
                        -target.clone()
                    };
                    absolute += if coefficient < rational(0, 1) {
                        -coefficient
                    } else {
                        coefficient
                    };
                }
                assert!(-absolute <= -target);
            }
        }
    }
    assert!(rational(1, 16) - &t_exact < rational(0, 1));
    let arbitrary = HeadFunctionalLpHint {
        row_duals: vec![0.125; old.rows.len()],
    };
    assert!(old.replay(&s, &arbitrary, 1.0, deadline()).unwrap() <= 1.0 / 16.0 - t);
    let legacy = super::super::proposals(&s, deadline()).unwrap();
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let cut = fit(&s, 3, &legacy, &mut budget).unwrap();
    assert_eq!(cut.positive, vec![(0, t), (1, t), (2, t)]);
    let new = Model::build(&s, deadline()).unwrap();
    assert_eq!(new.cone_cuts(), 1);
    let lower = new.replay(&s, &last_hint(&new), 1.0, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
}

#[test]
fn box_weighted_fit_recovers_useful_direction_excluded_by_nominal_pool_tie() {
    let heads = 66;
    let mut first = vec![0.0; 2 * heads];
    let mut last = vec![0.0; 2 * heads];
    // Sixty-four early columns all point along the narrow coordinate; the
    // useful wide direction is the sixty-fifth positive. Unweighted affine
    // cosine ranks every positive equally and pool64 loses that direction.
    for j in 0..64 {
        first[heads + j] = 2.0;
        last[j * 2] = 0.5;
    }
    first[64] = 2.0;
    last[128] = 0.5;
    first[65] = 1.0;
    first[heads + 65] = 1.0;
    last[130] = -1.0;
    let spec = InputHeadSpec::new(
        2,
        heads,
        first,
        vec![0.0; heads],
        last,
        vec![1.0 / 16.0, 0.0],
        [83; 32],
    )
    .unwrap();
    let s = Source::build(
        &spec,
        &[-1.0, -1.0 / 64.0],
        &[1.0, 1.0 / 64.0],
        0,
        1,
        deadline(),
    )
    .unwrap();
    let target_nominal = [1.0f64, 1.0];
    let mut nominal = (0..65)
        .map(|j| {
            let values = [
                f64::from(s.spec.first[j]),
                f64::from(s.spec.first[heads + j]),
            ];
            let dot = values
                .iter()
                .zip(target_nominal)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let norm = values.iter().map(|a| a * a).sum::<f64>().sqrt();
            (dot / norm, j)
        })
        .collect::<Vec<_>>();
    assert!(nominal.iter().all(|&(score, _)| score == 1.0));
    nominal.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    nominal.truncate(MAX_COLUMNS);
    assert!(!nominal.iter().any(|&(_, j)| j == 64));
    let legacy = super::super::proposals(&s, deadline()).unwrap();
    assert!(legacy
        .iter()
        .all(|cut| !cut.positive.iter().any(|&(j, _)| j == 64)));
    let mut budget = Budget::new(deadline(), MAX_WORK_CELLS);
    let cut = fit(&s, 65, &legacy, &mut budget).unwrap();
    assert!(cut.positive.iter().any(|&(j, a)| j == 64 && a == 0.5));
    assert_eq!(cut.positive.len(), 2);
    let model = manual_model(&s, &cut);
    let lower = model
        .replay(&s, &last_hint(&model), 1.0, deadline())
        .unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
}
