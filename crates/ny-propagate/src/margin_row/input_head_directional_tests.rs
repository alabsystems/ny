// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};
use ny_core::head_phase_cover_advice::{PhaseCoverLeafHint, PhaseCoverLpHint, PhaseCoverLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};

fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

fn cycles(bias: f32, permuted: bool) -> InputHeadSpec {
    let mut first = vec![0.0; 30 * 14];
    for head in 0..14 {
        let edge = if permuted { (head * 9 + 3) % 14 } else { head };
        let block = edge / 7 * 7;
        let input = |at| if permuted { (at * 11 + 5) % 30 } else { at };
        first[input(edge) * 14 + head] = 1.0;
        first[input(block + (edge + 1) % 7) * 14 + head] = -1.0;
    }
    InputHeadSpec::new(
        30,
        14,
        first,
        vec![0.0; 14],
        (0..14).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![bias, 0.0],
        [91; 32],
    )
    .unwrap()
}

fn source(bias: f32, permuted: bool) -> Source {
    Source::build(
        &cycles(bias, permuted),
        &[-1.0; 30],
        &[1.0; 30],
        0,
        1,
        end(),
    )
    .unwrap()
}

fn budget() -> Budget {
    Budget {
        deadline: end(),
        cells: 0,
        masks: 0,
    }
}

fn cycle_direction(block: usize) -> Direction {
    Direction {
        heads: (block * 7..block * 7 + 7).map(|head| (head, 1.0)).collect(),
    }
}

fn certified_cycles(s: &Source) -> Vec<CertifiedDirection<'_>> {
    let mut budget = budget();
    let directions = (0..2)
        .map(|block| certify(s, cycle_direction(block), &mut budget).unwrap())
        .collect();
    assert_eq!(budget.masks, 256);
    directions
}

struct SeparateCoverAdvice;
impl HeadPhaseAdvice for SeparateCoverAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_phase_cover_advice(&self) -> bool {
        true
    }
    fn propose_phase_cover(
        &self,
        request: &PhaseCoverLpRequest,
        _: Instant,
    ) -> Option<PhaseCoverLpHint> {
        let selected: Vec<_> = request.leaves[0]
            .rows
            .iter()
            .filter_map(|row| {
                row.coefficients
                    .iter()
                    .find(|&&(j, a)| j >= 30 && a == -1.0)
                    .map(|&(j, _)| j)
            })
            .collect();
        Some(PhaseCoverLpHint {
            leaves: request
                .leaves
                .iter()
                .map(|leaf| {
                    let mut row_duals = vec![0.0; request.rows.len() + leaf.rows.len()];
                    for (at, row) in request.rows.iter().enumerate() {
                        if let Some(&(_, a)) = row
                            .coefficients
                            .iter()
                            .find(|&&(j, a)| j >= 30 && a < 0.0 && !selected.contains(&j))
                        {
                            row_duals[at] = -1.0 / a;
                        }
                    }
                    for bit in 0..request.selected_heads {
                        row_duals[request.rows.len() + 2 * bit + 1] = 1.0;
                    }
                    PhaseCoverLeafHint {
                        mask: leaf.mask,
                        row_duals,
                    }
                })
                .collect(),
        })
    }
}

#[test]
fn independent_geometric_cuts_intersect_to_close_two_cycles() {
    let s = source(12.5, false);
    let directions = certified_cycles(&s);
    for direction in &directions {
        assert!(direction.lower <= -6.0 && direction.lower > -6.001);
    }
    let model = triangle::Model::from_root_directions(&s, &mut None, &directions, end()).unwrap();
    assert!(model.request().validate());
    let hint = native_hint(&model, &directions).unwrap();
    let bound = model.replay(&s, &hint, 1.0, end()).unwrap();
    assert!(bound > 0.49 && bound <= 0.5, "{bound}");
    let negated = HeadFunctionalLpHint {
        row_duals: hint.row_duals.iter().map(|v| -v).collect(),
    };
    assert_eq!(model.replay(&s, &negated, -1.0, end()), Some(bound));
    // Each specified complete scalar cover can close only one block.
    let before = s.baseline(end()).unwrap();
    let cover = phase_cover::refine(
        &s,
        before,
        PhaseCoverPolicy::new(7).unwrap(),
        end(),
        end(),
        &|| Some(Arc::new(SeparateCoverAdvice)),
    );
    assert!(
        cover.lower < -0.49 && cover.lower > -0.51,
        "{}",
        cover.lower
    );
    assert_eq!(cover.native_replayed, 128);
}

#[test]
fn support_selection_finds_both_cycles_under_head_and_input_permutation() {
    for permuted in [false, true] {
        let s = source(12.5, permuted);
        let mut work = budget();
        let proposed = proposals(&s, &mut work).unwrap();
        assert!(proposed.len() >= 2);
        assert_eq!(proposed[0].heads.len(), 7);
        assert_eq!(proposed[1].heads.len(), 7);
        for direction in proposed.into_iter().take(2) {
            let cut = certify(&s, direction, &mut work).unwrap();
            assert!(cut.lower > -6.001 && cut.lower <= -6.0);
        }
        let (report, _) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
        assert!(
            report.lower > 0.49 && report.lower <= 0.5,
            "{}",
            report.lower
        );
        assert!(report.cuts >= 2 && report.cuts <= MAX_CUTS);
        assert!(report.masks <= MAX_MASKS && report.work_cells <= MAX_CELLS);
    }
}

#[test]
fn unsafe_cycle_sibling_remains_open_on_original_source() {
    let spec = cycles(11.5, false);
    let outcome = verify_with_provider(
        &spec,
        &[-1.0; 30],
        &[1.0; 30],
        0,
        1,
        HeadGroupPolicy::new(3.0, 2, 4).unwrap(),
        end(),
        || None,
    );
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("unsafe Source authorized UNSAT")
    };
    assert!(report.lower.unwrap() <= -0.5);
    assert!(report.directional_cuts >= 2);
}

#[test]
fn geometry_excludes_output_bias_and_output_error_but_charges_every_first_error() {
    let a = source(12.5, false);
    let mut b = source(1000.0, false);
    b.output_error = 99.0;
    let ca = certify(&a, cycle_direction(0), &mut budget()).unwrap();
    let cb = certify(&b, cycle_direction(0), &mut budget()).unwrap();
    assert_eq!(ca.lower.to_bits(), cb.lower.to_bits());
    b.first_error[..7].fill(0.125);
    let charged = certify(&b, cycle_direction(0), &mut budget()).unwrap();
    assert!(charged.lower <= -6.5 && charged.lower > -6.501);
    let cuts = certified_cycles(&a);
    let ma = triangle::Model::from_root_directions(&a, &mut None, &cuts, end()).unwrap();
    let hint = native_hint(&ma, &cuts).unwrap();
    let ordinary = ma.replay(&a, &hint, 1.0, end()).unwrap();
    let mut changed = a.clone();
    changed.output_error += 0.25;
    let cuts = certified_cycles(&changed);
    let mb = triangle::Model::from_root_directions(&changed, &mut None, &cuts, end()).unwrap();
    let changed_bound = mb
        .replay(&changed, &native_hint(&mb, &cuts).unwrap(), 1.0, end())
        .unwrap();
    assert!((ordinary - changed_bound - 0.25).abs() < 1e-10);
}

#[test]
fn incomplete_last_algebraic_mask_never_mints_a_row_and_caps_are_shared() {
    let s = source(12.5, false);
    EXPIRE_AFTER_MASK.with(|fault| fault.set(Some(128)));
    let mut work = budget();
    assert!(certify(&s, cycle_direction(0), &mut work).is_none());
    assert_eq!(work.masks, 128);
    EXPIRE_AFTER_MASK.with(|fault| fault.set(None));
    let mut work = budget();
    for _ in 0..MAX_MASKS / 128 {
        assert!(certify(&s, cycle_direction(0), &mut work).is_some());
    }
    assert_eq!(work.masks, MAX_MASKS);
    assert!(certify(&s, cycle_direction(1), &mut work).is_none());
    let mut work = Budget {
        cells: MAX_CELLS,
        ..budget()
    };
    assert!(certify(&s, cycle_direction(0), &mut work).is_none());
}

#[test]
fn invalid_directions_foreign_sources_and_expired_certificates_refuse() {
    let s = source(12.5, false);
    for fault in 0..5 {
        let mut direction = cycle_direction(0);
        match fault {
            0 => direction.heads[1].0 = 0,
            1 => direction.heads[0].1 = -1.0,
            2 => direction.heads[0].1 = f64::NAN,
            3 => direction.heads[6].0 = 14,
            _ => direction.heads[0].1 = 0.3,
        }
        assert!(certify(&s, direction, &mut budget()).is_none());
    }
    let mut cut = certify(&s, cycle_direction(0), &mut budget()).unwrap();
    let clone = s.clone();
    assert!(cut.owned_row(&clone, end()).is_none());
    cut.deadline = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(cut.owned_row(&s, end()).is_none());
    let directions = certified_cycles(&s);
    let foreign = triangle::Model::build_phase_base(&clone, end())
        .unwrap()
        .into_phase_prefix();
    assert!(
        triangle::Model::from_root_directions(&s, &mut Some(foreign), &directions, end()).is_none()
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn joint_replay_prices_all_residuals_declines_bad_hints_and_never_exports_child_recipe() {
    let s = source(12.5, false);
    let directions = certified_cycles(&s);
    let model = triangle::Model::from_root_directions(&s, &mut None, &directions, end()).unwrap();
    let good_hint = native_hint(&model, &directions).unwrap();
    assert!(model
        .replay_recorded(&s, &good_hint, 1.0, end(), true)
        .is_none());
    for fault in 0..3 {
        let mut hint = good_hint.clone();
        match fault {
            0 => {
                hint.row_duals.pop();
            }
            1 => hint.row_duals[0] = f64::NAN,
            _ => hint.row_duals[0] = f64::INFINITY,
        }
        assert!(model.replay(&s, &hint, 1.0, end()).is_none());
    }
    let mut wrong = good_hint.clone();
    wrong.row_duals[0] = 1e80;
    assert!(model.replay(&s, &wrong, 1.0, end()).unwrap() < 0.0);
    wrong.row_duals.fill(0.0);
    assert!(model.replay(&s, &wrong, 1.0, end()).unwrap() <= s.baseline(end()).unwrap());
    let past = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert_eq!(commit_triangle_best(-1.0, 0.5, past, end()), -1.0);
    assert_eq!(commit_triangle_best(-1.0, 0.5, end(), past), -1.0);
}

#[test]
fn full_head_row_cap_keeps_every_scalar_triangle_and_only_replaces_optional_suffix() {
    let small = cycles(12.5, false);
    let mut first = vec![0.0; 30 * 512];
    for input in 0..30 {
        for head in 0..512 {
            first[input * 512 + head] = small.first[input * 14 + head % 14];
        }
    }
    let spec = InputHeadSpec::new(
        30,
        512,
        first,
        vec![0.0; 512],
        (0..512).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![1024.0, 0.0],
        [92; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 30], &[1.0; 30], 0, 1, end()).unwrap();
    let base = triangle::Model::build_phase_base(&s, end()).unwrap();
    let expected = base.request().rows.clone();
    assert_eq!(expected.len(), 1024);
    let mut directions = certified_cycles(&s);
    let mut extra = certified_cycles(&s);
    directions.append(&mut extra);
    let model = triangle::Model::from_root_directions(
        &s,
        &mut Some(base.into_phase_prefix()),
        &directions,
        end(),
    )
    .unwrap();
    assert_eq!(model.request().rows.len(), 1028);
    let model = triangle::Model::from_root_directions(
        &s,
        &mut Some(model.into_phase_prefix()),
        &directions,
        end(),
    )
    .unwrap();
    assert_eq!(model.request().rows.len(), 1028);
    for (actual, expected) in model.request().rows.iter().take(1024).zip(expected) {
        assert_eq!(actual.coefficients, expected.coefficients);
        assert_eq!(actual.lower, expected.lower);
    }
    assert!(model.request().validate());
}

#[test]
fn construction_refusal_preserves_the_entire_completed_prefix() {
    let s = source(12.5, false);
    let directions = certified_cycles(&s);
    let model = triangle::Model::from_root_directions(&s, &mut None, &directions, end()).unwrap();
    let expected = model.request().rows.clone();
    let mut prefix = Some(model.into_phase_prefix());
    let mut expired = certified_cycles(&s);
    expired[1].deadline = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(triangle::Model::from_root_directions(&s, &mut prefix, &expired, end()).is_none());
    assert!(prefix.is_some());
    let (_, rows) = prefix.unwrap().into_phase_parts(&s, 1056, end()).unwrap();
    assert_eq!(rows.len(), expected.len());
    for (actual, expected) in rows.iter().skip(28).zip(expected.iter().skip(28)) {
        assert_eq!(actual.lower.lo, expected.lower);
        assert_eq!(
            actual
                .coefficients
                .iter()
                .map(|&(j, a)| (j, a.lo))
                .collect::<Vec<_>>(),
            expected.coefficients
        );
    }
}

#[test]
fn weighted_signed_affine_geometry_encloses_exact_rational_vertex_maximum() {
    use num_rational::BigRational;
    use num_traits::{ToPrimitive, Zero};
    let rational = |value: f64| BigRational::from_float(value).unwrap();
    let first: Vec<f32> = (0_i32..3)
        .flat_map(|input| {
            (0_i32..8).map(move |head| (((input * 13 + head * 7 + 5) % 9) - 4) as f32 * 0.25)
        })
        .collect();
    let bias: Vec<f32> = (0_i32..8)
        .map(|head| ((head * 5 % 7) - 3) as f32 * 0.125)
        .collect();
    let spec = InputHeadSpec::new(
        3,
        8,
        first,
        bias,
        (0..8).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![100.0, 0.0],
        [93; 32],
    )
    .unwrap();
    let lo = [-1.0, -0.5, 0.25];
    let hi = [0.75, 1.0, 1.25];
    let s = Source::build(&spec, &lo, &hi, 0, 1, end()).unwrap();
    let direction = Direction {
        heads: (0..8)
            .map(|head| (head, [1.0, 0.5, 0.25][head % 3]))
            .collect(),
    };
    let mut work = budget();
    let cut = certify(&s, direction.clone(), &mut work).unwrap();
    assert_eq!(work.masks, 256);
    let mut maximum = BigRational::zero();
    for vertex in 0..8 {
        let mut mass = BigRational::zero();
        for &(head, tau) in &direction.heads {
            let mut value =
                rational(f64::from(spec.first_bias[head])) + rational(s.first_error[head]);
            for input in 0..3 {
                let x = if vertex & (1 << input) == 0 {
                    lo[input]
                } else {
                    hi[input]
                };
                value += rational(f64::from(spec.first[input * 8 + head])) * rational(x);
            }
            if value > BigRational::zero() {
                mass += rational(tau) * value;
            }
        }
        assert!(rational(-cut.lower) >= mass);
        maximum = maximum.max(mass);
    }
    assert!((-cut.lower - maximum.to_f64().unwrap()).abs() < 1e-10);
}

struct JointAdvice {
    calls: AtomicUsize,
}
impl HeadPhaseAdvice for JointAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        _: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut duals = vec![0.0; request.rows.len()];
        for (at, row) in request.rows.iter().enumerate() {
            if row.coefficients.len() == 7
                && row.coefficients.iter().all(|&(j, a)| j >= 30 && a == -1.0)
            {
                duals[at] = 1.0;
            }
        }
        Some(HeadFunctionalLpHint { row_duals: duals })
    }
}

#[test]
fn production_root_stage_replays_joint_advice_and_original_g3_default_policy() {
    let spec = cycles(12.5, false);
    let advice = Arc::new(JointAdvice {
        calls: AtomicUsize::new(0),
    });
    let policy = HeadGroupPolicy::new(3.0, 2, 4).unwrap();
    assert!(policy.directional_cuts());
    let result = verify_with_provider(&spec, &[-1.0; 30], &[1.0; 30], 0, 1, policy, end(), || {
        Some(advice.clone())
    });
    let InputHeadOutcome::Unsat(report, token) = result else {
        panic!("joint root stayed open")
    };
    assert_eq!(token.scope(), TokenScope::Instance);
    assert!(report.directional_before_lower.unwrap() < 0.0);
    assert!(report.directional_after_lower.unwrap() > 0.49);
    // A completed positive native joint replay no longer pays for advice it
    // cannot use. Actual AY and both sign conventions are exercised on the
    // separate unsafe/nondecisive feedback controls.
    assert!(report.directional_native_replayed > 0 || report.directional_ay_replayed > 0);
    assert!(report.directional_advice_calls <= MAX_ROUNDS);
    assert_eq!(
        advice.calls.load(Ordering::Relaxed),
        report.advice_calls + report.directional_advice_calls
    );
}
