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
fn rational(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn spec(b: f32) -> InputHeadSpec {
    let mut first = vec![0.0; 5 * 6];
    for j in 0..5 {
        first[j * 6 + j] = 1.0;
        first[((j + 1) % 5) * 6 + j] = -1.0;
    }
    let mut bias = vec![0.0; 6];
    bias[5] = 1.0;
    let mut final_weight = vec![0.0; 12];
    for j in 0..5 {
        final_weight[j * 2] = -1.0;
    }
    final_weight[10] = 1.0;
    InputHeadSpec::new(5, 6, first, bias, final_weight, vec![b, 0.0], [91; 32]).unwrap()
}
fn source(b: f32) -> Source {
    Source::build(&spec(b), &[-1.0; 5], &[1.0; 5], 0, 1, deadline()).unwrap()
}
fn group() -> Group {
    Group {
        heads: (0..5).map(|j| (j, 1.0)).collect(),
    }
}
fn old(s: &Source) -> Model<'_> {
    Model::build_with_large_groups(s, true, true, true, false, deadline()).unwrap()
}
fn new(s: &Source) -> Model<'_> {
    Model::build(s, deadline()).unwrap()
}
fn same_row(a: &OwnedRow, b: &OwnedRow) -> bool {
    a.lower.lo.to_bits() == b.lower.lo.to_bits()
        && a.lower.hi.to_bits() == b.lower.hi.to_bits()
        && a.coefficients.len() == b.coefficients.len()
        && a.coefficients
            .iter()
            .zip(&b.coefficients)
            .all(|(&(i, a), &(j, b))| {
                i == j && a.lo.to_bits() == b.lo.to_bits() && a.hi.to_bits() == b.hi.to_bits()
            })
}
fn hint(m: &Model<'_>, p: f64) -> HeadFunctionalLpHint {
    let at = m
        .rows
        .iter()
        .position(|row| {
            row.coefficients.len() == 5
                && row
                    .coefficients
                    .iter()
                    .enumerate()
                    .all(|(at, &(j, a))| j == 5 + at && a.lo == a.hi && a.hi < 0.0)
        })
        .unwrap();
    let mut row_duals = vec![0.0; m.rows.len()];
    row_duals[at] = p;
    HeadFunctionalLpHint { row_duals }
}
fn exact(x: &[i64; 5], b: i64) -> BigRational {
    (0..5).fold(rational(b, 16) + rational(1, 1), |v, j| {
        v - rational((x[j] - x[(j + 1) % 5]).max(0), 1)
    })
}

#[test]
fn every_proper_cycle_projection_has_two_actual_opposite_corner_witnesses() {
    // Every proper edge subset is a forest. This is a constructive convex
    // witness for ALL valid rows on that subset, not just the proposed rows.
    for selected in 1usize..31 {
        let coloring = (0usize..32)
            .find(|&bits| {
                (0..5).all(|j| {
                    selected & (1 << j) == 0 || ((bits >> j) & 1) != ((bits >> ((j + 1) % 5)) & 1)
                })
            })
            .unwrap();
        let x: Vec<i64> = (0..5)
            .map(|j| if coloring & (1 << j) == 0 { -1 } else { 1 })
            .collect();
        for j in 0..5 {
            assert_eq!(x[j] + (-x[j]), 0);
            if selected & (1 << j) != 0 {
                let y = x[j] - x[(j + 1) % 5];
                assert_eq!(y.max(0) + (-y).max(0), 2);
            }
        }
    }
    assert!(
        (0usize..32).all(|bits| (0..5).any(|j| ((bits >> j) & 1) == ((bits >> ((j + 1) % 5)) & 1)))
    );
}

#[test]
fn all_actual_old_triangle_cone_and_small_group_rows_admit_the_fractional_point() {
    let s = source(49.0 / 16.0);
    let m = old(&s);
    assert!(m.negative_groups() > 0 && m.negative_group_masks() <= 128);
    assert_eq!(m.large_negative_groups(), 0);
    let point = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    for (&x, b) in point.iter().zip(&m.boxes) {
        assert!(b.lo <= x && x <= b.hi);
    }
    for row in &m.rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |v, &(j, a)| {
            v.add(a.mul(I::p(point[j])).unwrap()).unwrap()
        });
        assert!(
            lhs.lo >= row.lower.hi,
            "the actual owned old row admits the point"
        );
    }
    for row in &m.request.rows {
        let lhs = row
            .coefficients
            .iter()
            .map(|&(j, a)| a * point[j])
            .sum::<f64>();
        assert!(
            lhs >= row.lower,
            "the actual exported old proposal row admits the point"
        );
    }
    assert_eq!(
        rational(49, 16) + rational(1, 1) - rational(5, 1),
        rational(-15, 16)
    );
    for p in [0.0, 0.25, 1.0, 2.0] {
        let dual = HeadFunctionalLpHint {
            row_duals: vec![p; m.rows.len()],
        };
        assert!(m.replay(&s, &dual, 1.0, deadline()).unwrap() <= -15.0 / 16.0);
    }
}

#[test]
fn grown_all32_mask_row_preserves_every_old_row_and_closes_the_cycle() {
    let s = source(49.0 / 16.0);
    let old = old(&s);
    let m = new(&s);
    assert_eq!(m.large_negative_groups(), 1);
    assert_eq!(m.large_negative_group_masks(), 32);
    assert_eq!(m.rows.len(), old.rows.len() + 1);
    assert!(old.rows.iter().zip(&m.rows).all(|(a, b)| same_row(a, b)));
    let lower = m.replay(&s, &hint(&m, 1.0), 1.0, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
    for bits in 0..32 {
        let x = std::array::from_fn(|j| if bits & (1 << j) == 0 { -1 } else { 1 });
        assert!(BigRational::from_float(lower).unwrap() <= exact(&x, 49));
    }
}

#[derive(Clone, Copy)]
enum Fault {
    Valid,
    Missing,
    Nan,
    Panic,
}
struct Advice {
    sign: f64,
    fault: Fault,
    calls: AtomicUsize,
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
        assert_eq!(r.feature_count, 5);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut row_duals = vec![0.0; r.rows.len()];
        if r.bounds.len() == 6 {
            self.legacy.fetch_add(1, Ordering::SeqCst);
        } else {
            assert_eq!(r.bounds.len(), 11);
            self.triangle.fetch_add(1, Ordering::SeqCst);
            let at = r
                .rows
                .iter()
                .position(|row| {
                    row.coefficients.len() == 5
                        && row
                            .coefficients
                            .iter()
                            .enumerate()
                            .all(|(at, &(j, a))| j == 5 + at && a < -0.99)
                })
                .unwrap();
            match self.fault {
                Fault::Valid => row_duals[at] = self.sign,
                Fault::Missing => {
                    row_duals.pop();
                }
                Fault::Nan => row_duals[at] = f64::NAN,
                Fault::Panic => panic!("named untrusted advice fault"),
            }
        }
        Some(HeadFunctionalLpHint { row_duals })
    }
}
fn outcome(b: f32, sign: f64, fault: Fault) -> (InputHeadOutcome, Arc<Advice>) {
    let advice = Arc::new(Advice {
        sign,
        fault,
        calls: AtomicUsize::new(0),
        legacy: AtomicUsize::new(0),
        triangle: AtomicUsize::new(0),
    });
    let result = verify_with_provider(
        &spec(b),
        &[-1.0; 5],
        &[1.0; 5],
        0,
        1,
        // Keep the original large-group callback and injected-fault boundary.
        HeadGroupPolicy::new(1.0, 1, 1)
            .unwrap()
            .with_directional_cuts(false),
        deadline(),
        || Some(Arc::clone(&advice) as Arc<dyn HeadPhaseAdvice>),
    );
    (result, advice)
}

#[test]
fn real_singleton_instance_authorizer_uses_same_two_mock_advice_calls_both_signs() {
    // This is real Source replay/publication with a deterministic mock hint.
    // It does NOT claim AY executed or returned this multiplier.
    for sign in [1.0, -1.0] {
        let (InputHeadOutcome::Unsat(report, token), a) = outcome(49.0 / 16.0, sign, Fault::Valid)
        else {
            panic!("complete source cycle row should close");
        };
        assert_eq!(token.scope(), TokenScope::Instance);
        assert_eq!(token.classes(), &[1]);
        assert!(token.covers_all(&[1]) && !token.covers_all(&[0, 1]));
        assert!(report.lower.unwrap() > 1.0 / 32.0 && report.lower.unwrap() <= 1.0 / 16.0);
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.replayed_candidates, 5);
        assert_eq!(report.triangle_variables, 11);
        assert_eq!(report.triangle_advice_calls, 1);
        assert_eq!(report.triangle_large_negative_groups, 1);
        assert_eq!(report.triangle_large_negative_group_masks, 32);
        assert_eq!(a.calls.load(Ordering::SeqCst), 2);
        assert_eq!(a.legacy.load(Ordering::SeqCst), 1);
        assert_eq!(a.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn actual_cycle_tie_and_negative_instances_stay_open_after_mock_provider_arrival() {
    for b in [3.0, 47.0 / 16.0] {
        let (InputHeadOutcome::Unknown(report), a) = outcome(b, 1.0, Fault::Valid) else {
            panic!("unsafe cycle cannot publish");
        };
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.triangle_large_negative_group_masks, 32);
        assert!(report.lower.unwrap() <= f64::from(b) - 3.0 && report.lower.unwrap() <= 0.0);
        assert_eq!(a.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn malformed_missing_or_panicking_mock_duals_never_authorize_the_cycle() {
    for fault in [Fault::Missing, Fault::Nan, Fault::Panic] {
        let (InputHeadOutcome::Unknown(report), a) = outcome(49.0 / 16.0, 1.0, fault) else {
            panic!("faulty advice has no authority");
        };
        assert_eq!(report.advice_calls, 2);
        assert_eq!(report.replayed_candidates, 3);
        // These counters describe admitted replay, not the completed row
        // seen by the provider before it deliberately injects its fault.
        assert_eq!(report.triangle_large_negative_groups, 0);
        assert_eq!(report.triangle_large_negative_group_masks, 0);
        assert_eq!(report.triangle_replayed_candidates, 0);
        assert_eq!(report.triangle_classes_improved, 0);
        let before = report.triangle_before_lower.unwrap();
        let after = report.triangle_after_lower.unwrap();
        assert_eq!(before.to_bits(), after.to_bits());
        assert_eq!(report.lower.unwrap().to_bits(), before.to_bits());
        assert!(report.lower.unwrap().is_finite() && report.lower.unwrap() <= 0.0);
        assert_eq!(a.calls.load(Ordering::SeqCst), 2);
        assert_eq!(a.triangle.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn shared256_mask_cap_includes_old_and_partial_prefix_work() {
    let s = source(49.0 / 16.0);
    let g = group();
    let mut b = Budget::new(deadline(), 224, 0);
    assert!(owned_group_row(&s, &g, &mut b).is_some());
    assert_eq!(b.report.complete_masks, 32);
    let mut full = Budget::new(deadline(), 225, 0);
    assert!(owned_group_row(&s, &g, &mut full).is_none());
    assert_eq!(full.report.complete_masks, 0);
    assert!(full.report.exhausted);
    let mut old_full = Budget::new(deadline(), 256, 0);
    assert!(owned_group_row(&s, &g, &mut old_full).is_none());
}

#[test]
fn shared_work_cap_refuses_without_renewing_the_old_allowance() {
    let s = source(49.0 / 16.0);
    let mut b = Budget::new(deadline(), 0, SHARED_WORK_CELLS);
    assert!(owned_group_row(&s, &group(), &mut b).is_none());
    assert!(b.report.exhausted);
    assert_eq!(b.report.complete_masks, 0);
    let (rows, r) = rows(&s, &[], 0, SHARED_WORK_CELLS + 1, deadline());
    assert!(rows.is_empty() && r.exhausted);
}

#[test]
fn deterministic_expiry_after_real_masks_never_emits_a_prefix_row() {
    let s = source(49.0 / 16.0);
    for at in [1, 31, 32] {
        EXPIRE_AFTER_MASK.with(|f| f.set(Some(at)));
        let mut b = Budget::new(deadline(), 0, 0);
        assert!(owned_group_row(&s, &group(), &mut b).is_none());
        assert_eq!(b.report.complete_masks, at);
        assert!(b.report.expired);
        EXPIRE_AFTER_MASK.with(|f| f.set(None));
    }
}

#[test]
fn unfinished_large_group_keeps_every_completed_old_model_row() {
    let s = source(49.0 / 16.0);
    let old = old(&s);
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(31)));
    let m = new(&s);
    EXPIRE_AFTER_MASK.with(|f| f.set(None));
    assert_eq!(m.large_negative_groups(), 0);
    assert_eq!(m.rows.len(), old.rows.len());
    assert!(m.rows.iter().zip(&old.rows).all(|(a, b)| same_row(a, b)));
}

#[test]
fn last_original_input_feature_is_present_in_all32_mask_supports() {
    let mut authored = spec(49.0 / 16.0);
    for j in 0..5 {
        authored.first[4 * 6 + j] += 1.0;
    }
    let s = Source::build(&authored, &[-1.0; 5], &[1.0; 5], 0, 1, deadline()).unwrap();
    let row = owned_group_row(&s, &group(), &mut Budget::new(deadline(), 0, 0)).unwrap();
    assert!(row.lower.lo <= -5.0);
    let mut m = Model::build_with_large_groups(&s, false, false, false, false, deadline()).unwrap();
    m.request.rows.push(HeadFunctionalLpRow {
        coefficients: row.coefficients.iter().map(|&(j, a)| (j, a.lo)).collect(),
        lower: row.lower.lo,
    });
    m.rows.push(row);
    assert!(m.replay(&s, &hint(&m, 1.0), 1.0, deadline()).unwrap() <= -15.0 / 16.0);
}

#[test]
fn full_last_head_unequal_q_and_overdrawn_mass_remain_in_the_residual() {
    // Authenticate unequal Q through the original authored output row.
    let mut authored = spec(49.0 / 16.0);
    authored.final_weight[8] = -2.0;
    let s = Source::build(&authored, &[-1.0; 5], &[1.0; 5], 0, 1, deadline()).unwrap();
    let mut m = Model::build_with_large_groups(&s, false, false, false, false, deadline()).unwrap();
    let row = owned_group_row(&s, &group(), &mut Budget::new(deadline(), 0, 0)).unwrap();
    m.request.rows.push(HeadFunctionalLpRow {
        coefficients: row.coefficients.iter().map(|&(j, a)| (j, a.lo)).collect(),
        lower: row.lower.lo,
    });
    m.rows.push(row);
    for p in [0.0, 0.25, 1.0, 2.0] {
        let lower = m.replay(&s, &hint(&m, p), 1.0, deadline()).unwrap();
        // A real corner with the LAST edge active has sumReLU4 plus last2.
        let corner = [-1, 1, -1, 1, 1];
        assert!(corner[4] - corner[0] > 0);
        let actual = exact(&corner, 49) - rational(2, 1);
        assert!(BigRational::from_float(lower).unwrap() <= actual && lower < 0.0);
    }
}

#[test]
fn independent_first_layer_and_output_error_uncertainty_cannot_cancel() {
    let mut s = source(49.0 / 16.0);
    for j in 0..5 {
        s.first_error[j] += 0.5;
        s.y[j] = s.y[j].add(I { lo: -0.5, hi: 0.5 }).unwrap();
    }
    s.output_error += 0.5;
    s.bias = I {
        lo: 3.0,
        hi: 49.0 / 16.0,
    };
    let m = new(&s);
    assert!(m.replay(&s, &hint(&m, 1.0), 1.0, deadline()).unwrap() < 0.0);
}

#[test]
fn tiny_mass_times_subnormal_weight_is_charged_before_large_input_support() {
    let mut authored = spec(49.0 / 16.0);
    for k in 0..5 {
        for j in 0..5 {
            authored.first[k * 6 + j] = f32::from_bits(1);
        }
    }
    let radius = 2.0f64.powi(120);
    let s = Source::build(&authored, &[-radius; 5], &[radius; 5], 0, 1, deadline()).unwrap();
    let tau = f64::from_bits(1);
    let g = Group {
        heads: (0..5).map(|j| (j, tau)).collect(),
    };
    let row = owned_group_row(&s, &g, &mut Budget::new(deadline(), 0, 0)).unwrap();
    assert!(row.lower.lo < -1e-300 && row.lower.lo < -tau);
}

#[test]
fn invalid_duplicate_missing_last_dimensions_and_foreign_generation_refuse() {
    let s = source(49.0 / 16.0);
    for heads in [
        vec![(0, 1.0); 5],
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (6, 1.0)],
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, f64::NAN)],
    ] {
        assert!(
            owned_group_row(&s, &Group { heads }, &mut Budget::new(deadline(), 0, 0)).is_none()
        );
    }
    let mut short = source(49.0 / 16.0);
    short.lo.pop();
    assert!(owned_group_row(&short, &group(), &mut Budget::new(deadline(), 0, 0)).is_none());
    let mut short = source(49.0 / 16.0);
    short.q.pop();
    assert!(owned_group_row(&short, &group(), &mut Budget::new(deadline(), 0, 0)).is_none());
    let m = new(&s);
    let foreign = source(49.0 / 16.0);
    assert!(m
        .replay(&foreign, &hint(&m, 1.0), 1.0, deadline())
        .is_none());
}

#[test]
fn grow_search_is_generic_and_no_completed_old_four_head_seed_means_no_new_row() {
    let s = source(49.0 / 16.0);
    let (none, report) = rows(&s, &[], 0, 0, deadline());
    assert!(none.is_empty() && report.complete_masks == 0);
    let old = old(&s);
    let (grown, report) = rows(&s, &old.rows, old.negative_group_masks(), 0, deadline());
    assert_eq!(grown.len(), 1);
    assert_eq!(report.complete_masks, 32);
    assert!(report.rank_trials <= MAX_RANK_TRIALS);
    assert!(grown.iter().all(|(_, masks)| *masks == 32));
}

#[test]
fn retained_row_tags_are_required_only_when_producing_a_recipe() {
    let s = source(49.0 / 16.0);
    let mut m = new(&s);
    let hint = hint(&m, 1.0);
    assert!(m
        .replay_recorded(&s, &hint, 1.0, deadline(), true)
        .is_some());
    m.row_tags.pop();
    // Provenance is not used by the complete numerical-only weak dual.
    assert!(m.replay(&s, &hint, 1.0, deadline()).is_some());
    // Missing provenance can never mint an imported/widened residual recipe.
    assert!(m
        .replay_recorded(&s, &hint, 1.0, deadline(), true)
        .is_none());
}
