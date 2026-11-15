// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual checker controls; all discovery claims below use the public entry.
use super::super::default_fixture::{end, lp, q, rows_hold, solve};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

fn repeated_source(bias: f64, positive: usize) -> HullMarginProblem {
    let h = positive + 16;
    let mut output = vec![0.; 2 * h];
    for j in 0..h {
        let coefficient = if j < positive {
            1. / positive as f64
        } else {
            -1. / 16.
        };
        output[j] = coefficient.next_up().next_up();
    }
    let p = HullMarginProblem::new(
        vec![1.; h],
        vec![0.; h],
        1,
        &output,
        &[bias.next_up().next_up(), 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    assert_eq!(p.d, bias);
    assert!(p.c[..positive].iter().all(|c| *c == 1. / positive as f64));
    assert!(p.c[positive..].iter().all(|c| *c == -1. / 16.));
    p
}
fn prepared(
    p: &HullMarginProblem,
    plan: Plan,
) -> (
    HullRootMipCuts<'_>,
    Plan,
    Budget,
    certificate::Checked,
    usize,
) {
    let m = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let cuts = prepare(p, m, lower, end(), &mut || true).unwrap();
    plan.valid(&cuts).unwrap();
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    (cuts, plan, b, checked, cost)
}
fn repeated_plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: (1..9).collect(),
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.25],
                alpha: (0..256usize)
                    .map(|mask| 0.25 + mask.count_ones() as f64 / 16.)
                    .collect(),
            },
            Component {
                negative: (9..17).collect(),
                positive: vec![],
                positive_units: vec![],
                g: vec![-0.25],
                alpha: vec![],
            },
        ],
        leftover: vec![],
    }
}
#[test]
fn preserved_masks_supplied_halfshare_has_strict_checked_safe_unsafe_and_tie_controls() {
    for bias in [0.125, -0.125, 0.] {
        let p = repeated_source(bias, 1);
        let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p, repeated_plan());
        let before = checked.root_lower;
        assert!((before - (bias - 0.25)).abs() < 1e-9);
        let work = b.state.lock().unwrap().work;
        let checks = b.state.lock().unwrap().checks;
        let negative: Vec<_> = plan.components.iter().map(|c| c.negative.clone()).collect();
        assert!(super::super::run(
            &cuts,
            &mut plan,
            &mut checked,
            &mut cost,
            f64::NEG_INFINITY,
            &mut b,
            &mut || true
        ));
        let r = b.state.lock().unwrap().root_coupling.clone();
        assert!(r.selected && r.completed_supports == 1, "{r:?}");
        assert_eq!(r.stop_reason, "preserved-root-improved");
        assert_eq!((r.old_masks, r.new_masks), (512, 512));
        assert_eq!(r.parent_count, 2);
        assert_eq!(b.state.lock().unwrap().work, work + LIMIT);
        assert_eq!(b.state.lock().unwrap().checks, checks + 1);
        assert_eq!(r.producer_attempts, 1);
        assert_eq!(r.support_attempts, 1);
        assert_eq!(r.checker_slots, 1);
        assert!(r.used_work_units <= LIMIT && b.payload_bytes().unwrap() <= MAX_BYTES);
        assert!(cost <= r.quoted_check_work);
        assert!(checked.root_lower > bias - 1e-9 && checked.root_lower <= bias);
        assert!(checked.root_lower > before + 0.249999);
        for (c, old) in plan.components.iter().zip(&negative) {
            assert_eq!(&c.negative, old);
            assert_eq!(c.positive, [0]);
            assert_eq!(c.positive_units, [1]);
            assert_eq!(c.g, [0.]);
        }
        for x in [-1_f64, 0., 1.] {
            let actual = bias + x.max(0.) - 16. * (x.max(0.) / 16.);
            assert!(checked.root_lower <= actual);
        }
        assert_eq!(checked.root_lower > 0., bias > 0.);
        let ticket = b.root_coupling.as_ref().unwrap();
        assert!(ticket.claim_checker(&cuts, &b).is_none());
    }
}

#[test]
fn preserved_saturated_exchange_keeps_both_sixteen_slot_caps_and_every_original_mask() {
    let p = repeated_source(0., 32);
    let plan = Plan {
        components: (0..2)
            .map(|part| Component {
                negative: (32 + part * 8..40 + part * 8).collect(),
                positive: (part * 16..(part + 1) * 16).collect(),
                positive_units: vec![2; 16],
                g: vec![if part == 0 { 0.25 } else { -0.25 }],
                alpha: vec![0.5; 256 * 16],
            })
            .collect(),
        leftover: vec![],
    };
    let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p, plan);
    let old = checked.root_lower;
    let old_cost = cost;
    assert!(super::super::run(
        &cuts,
        &mut plan,
        &mut checked,
        &mut cost,
        f64::NEG_INFINITY,
        &mut b,
        &mut || true
    ));
    let r = b.state.lock().unwrap().root_coupling.clone();
    assert!(r.selected && r.completed_supports == 1, "{r:?}");
    assert!(checked.root_lower > old && checked.root_lower <= 0.);
    assert_eq!(cost, old_cost);
    assert_eq!((r.old_masks, r.new_masks), (512, 512));
    assert_eq!(plan.components[0].positive[0], 16);
    assert_eq!(plan.components[1].positive[0], 0);
    for (index, c) in plan.components.iter().enumerate() {
        assert_eq!(c.positive.len(), 16);
        assert_eq!(c.positive_units, [2; 16]);
        assert_eq!(
            c.negative,
            (32 + index * 8..40 + index * 8).collect::<Vec<_>>()
        );
    }
    plan.valid(&cuts).unwrap();
}

#[test]
fn preserved_cancellation_retains_exact_baseline_and_irreversible_original_debt() {
    let p = repeated_source(0.125, 1);
    let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p, repeated_plan());
    let old_bits = witness(&plan);
    let old_lower = checked.root_lower.to_bits();
    let old_cost = cost;
    let work = b.state.lock().unwrap().work;
    let checks = b.state.lock().unwrap().checks;
    let state = b.state.clone();
    let mut interrupted = false;
    assert!(super::super::run(
        &cuts,
        &mut plan,
        &mut checked,
        &mut cost,
        f64::NEG_INFINITY,
        &mut b,
        &mut || {
            if state.lock().unwrap().root_coupling.line_mask_attempts > 0 {
                interrupted = true;
                false
            } else {
                true
            }
        }
    ));
    assert!(interrupted);
    assert_eq!(witness(&plan), old_bits);
    assert_eq!(checked.root_lower.to_bits(), old_lower);
    assert_eq!(cost, old_cost);
    let r = b.state.lock().unwrap().root_coupling.clone();
    assert_eq!(b.state.lock().unwrap().work, work + LIMIT);
    assert_eq!(b.state.lock().unwrap().checks, checks + 1);
    assert_eq!(r.support_attempts, 0);
    assert!(!r.selected);
    assert!(b.state.lock().unwrap().reserved >= 2 * cost);
}

#[test]
fn preserved_admission_binds_real_vec_capacities_and_rejects_foreign_source() {
    let p = repeated_source(0.125, 1);
    let (cuts, mut plan, mut b, _checked, cost) = prepared(&p, repeated_plan());
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    assert!(ticket.preserved_mode);
    ticket.begin(&cuts, &b).unwrap();
    assert!(ticket.matches_plan(&plan, cost));
    let before = ticket.remaining(&cuts, &b).unwrap();
    assert!(ticket.claim_selection(&cuts, &b, &plan, cost).is_none());
    assert!(ticket.claim_checker(&cuts, &b).is_none());
    assert!(!ticket.state.lock().unwrap().selection);
    assert_eq!(ticket.remaining(&cuts, &b), Some(before));
    plan.components[0].alpha.reserve(2048);
    assert!(!ticket.matches_plan(&plan, cost));
    let foreign = repeated_source(0.125, 1);
    let (other, _, other_b, _, _) = prepared(&foreign, repeated_plan());
    assert!(!ticket.owned(&other, &other_b));
    assert!(ticket.claim_checker(&other, &other_b).is_none());
    let ordinary = test_fixture::source(0.125);
    let (ordinary_cuts, ordinary_plan, mut ordinary_budget, _, ordinary_cost) =
        test_fixture::baseline(&ordinary);
    let (ordinary_ticket, _) = admit(
        &ordinary_cuts,
        &ordinary_plan,
        ordinary_cost,
        &mut ordinary_budget,
    )
    .unwrap();
    assert!(!ordinary_ticket.preserved_mode);
    ordinary_ticket
        .begin(&ordinary_cuts, &ordinary_budget)
        .unwrap();
    let mut polls = 0;
    assert!(choose(
        &ordinary_cuts,
        &ordinary_plan,
        &ordinary_budget,
        &ordinary_ticket,
        ordinary_cost,
        &mut || {
            polls += 1;
            true
        }
    )
    .is_none());
    assert_eq!(polls, 0);
    assert!(!ordinary_ticket.state.lock().unwrap().selection);
    assert!(ordinary_ticket
        .claim_checker_mode(&ordinary_cuts, &ordinary_budget, true)
        .is_none());
}

const D: usize = 8;
const H: usize = 10;
const ETA: f64 = 1. / 64.;
struct Cycle {
    w: Vec<Vec<f64>>,
    bias: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Cycle {
    fn new(bias: f64) -> Self {
        let mut first = vec![0.; D * H];
        for edge in 0..7 {
            first[(edge + 1) * D + edge] = 1.;
            first[(edge + 1) * D + (edge + 1) % 7] = -1.;
        }
        for j in [0, 8, 9] {
            first[j * D] = 1.;
            first[j * D + 1] = -1.;
            first[j * D + 7] = 8.;
        }
        let mut output = vec![0.; 2 * H];
        output[0] = 2. * ETA;
        output[1..8].fill(-1.);
        output[8..].iter_mut().take(2).for_each(|v| *v = -ETA);
        let mut bounds = vec![Bound::new(-2., 2.); H];
        for j in [0, 8, 9] {
            bounds[j] = Bound::new(-10., 10.);
        }
        Self {
            w: vec![first, output],
            bias: vec![vec![0.; H], vec![bias, 0.]],
            dims: vec![D, H, 2],
            bounds: vec![bounds],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.w[0].clone(),
            self.bias[0].clone(),
            D,
            &self.w[1],
            &self.bias[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; D],
            vec![1.; D],
        )
        .unwrap()
    }
    fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.w,
            &self.bias,
            &self.dims,
            &[Bound::new(-1., 1.); D],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    fn parts(&self) -> MipParts {
        let mut parts = self.original();
        assert!(parts
            .problem
            .relax_binary_integrality(&[parts.binary_vars[0]]));
        parts.binary_vars.remove(0);
        parts.binary_widths.remove(0);
        parts
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.w,
            biases: &self.bias,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
    fn point(&self, parts: &MipParts, input: &[Q], zero_mask: usize) -> Vec<Q> {
        let original = self.original();
        let heads = binding::head_columns(&self.encoding(), &original).unwrap();
        let mut values = vec![Q::zero(); parts.num_cols];
        for i in 0..D {
            values[parts.input_vars[i].0] = input[i].clone();
        }
        for j in 0..H {
            let z = (0..D)
                .map(|i| q(self.w[0][j * D + i]) * &input[i])
                .sum::<Q>();
            values[D + j] = z.clone();
            values[heads[j].0] = z.clone().max(Q::zero());
            let active = if z.is_zero() {
                (zero_mask >> j) & 1
            } else {
                usize::from(z > Q::zero())
            };
            values[original.binary_vars[j].0] = q(active as f64);
        }
        for out in 0..2 {
            values[parts.output_vars[out].0] = q(self.bias[1][out])
                + (0..H)
                    .map(|j| q(self.w[1][out * H + j]) * &values[heads[j].0])
                    .sum::<Q>();
        }
        values
    }
}
fn default_cycle(bias: f64) -> (Cycle, MipParts, HullMipCutReport) {
    let f = Cycle::new(bias);
    let p = f.source();
    // Real public discovery: no Plan, grouping, alpha, pair or captured point.
    let attempt = propose_root_mip_cuts_with_root_coupling_attempt(&p, end(), Duration::ZERO);
    assert!(
        attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked,
        "{:?}",
        attempt.primal
    );
    let cuts = attempt.cuts.expect("actual root-policy owner");
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert!(
        cuts.multipliers.alpha[0].is_finite() && (0. ..=1.).contains(&cuts.multipliers.alpha[0]),
        "actual LP-derived positive alpha"
    );
    let r = cuts.signed_report.as_ref().unwrap().root_coupling.clone();
    assert!(r.admitted && r.selected, "{r:?}");
    assert_eq!(r.stop_reason, "preserved-root-improved");
    assert_eq!((r.parent_count, r.old_masks, r.new_masks), (2, 258, 258));
    assert_eq!(r.selected_negatives, 8);
    assert_eq!(r.tail_negatives, 0);
    assert_eq!((r.positive_heads, r.recipient_positive_heads), (1, 1));
    assert_eq!(
        (
            r.producer_attempts,
            r.support_attempts,
            r.completed_supports,
            r.checker_slots
        ),
        (1, 1, 1, 1)
    );
    // The bound d-6-1/16 holds for every original whole-allocation alpha/g,
    // independent of which optimal LP dual is returned.
    assert!(
        r.before_global_lower.unwrap() <= bias - 6. - 1. / 16. + 1e-8,
        "{r:?}"
    );
    assert!(r.after_global_lower.unwrap() > bias - 6. - 1e-8, "{r:?}");
    assert!(r.after_global_lower.unwrap() <= bias - 6., "{r:?}");
    let mut parts = f.parts();
    let old_rows = parts.problem.num_rows();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    assert!(report.cuts > 0 && parts.problem.num_rows() > old_rows);
    assert_eq!(report.signed.as_ref().unwrap().root_coupling, r);
    assert!(report.signed.as_ref().unwrap().admitted_bytes <= MAX_BYTES);
    assert!(report.signed.as_ref().unwrap().work_units <= MAX_WORK);
    if bias > 6. {
        let proof = cuts
            .signed_root_proof(end())
            .expect("actual original Source replay");
        assert!(proof.verified() && proof.lower() > 0.);
    }
    (f, parts, report)
}
#[test]
fn preserved_default_cycle_discovers_original_large_group_and_closes_actual_safe_model() {
    let (f, parts, report) = default_cycle(6. + 1. / 32.);
    let property = f.encoding().unsafe_row.0;
    let original = f.parts();
    assert!(lp(&original.problem, &original, property) < 0.);
    let after = lp(&parts.problem, &parts, property);
    assert!(after > 0. && after <= 1. / 32. + 1e-7);
    assert!(
        after + 1e-7
            >= report
                .signed
                .as_ref()
                .unwrap()
                .root_coupling
                .after_global_lower
                .unwrap()
    );
    for mask in 0..1usize << D {
        let input: Vec<_> = (0..D)
            .map(|i| q(if mask & (1 << i) == 0 { -1. } else { 1. }))
            .collect();
        for zeros in [0, (1usize << H) - 1] {
            rows_hold(
                &parts.problem,
                &f.point(&parts, &input, zeros),
                Some(property),
            );
        }
    }
    let result = solve(parts);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}
fn actual_sat(bias: f64) {
    let (f, parts, report) = default_cycle(bias);
    let expected = bias - 6.;
    assert!(
        report
            .signed
            .as_ref()
            .unwrap()
            .root_coupling
            .after_global_lower
            .unwrap()
            <= expected
    );
    let input: Vec<_> = [1., -1., 1., -1., 1., -1., 1., 0.]
        .into_iter()
        .map(q)
        .collect();
    for zeros in [0, (1usize << H) - 1] {
        let point = f.point(&parts, &input, zeros);
        rows_hold(&parts.problem, &point, None);
        assert_eq!(point[parts.output_vars[0].0], q(expected));
    }
    match solve(parts) {
        crate::MipResult::Sat { input_values, .. } => {
            let original = f.original();
            let input: Vec<_> = input_values.into_iter().map(q).collect();
            let point = f.point(&original, &input, 0);
            rows_hold(&original.problem, &point, None);
            assert!(point[original.output_vars[0].0] <= Q::zero());
        }
        other => panic!("{other:?}"),
    }
}
#[test]
fn preserved_default_cycle_keeps_original_unsafe_witness() {
    actual_sat(6. - 1. / 32.);
}
#[test]
fn preserved_default_cycle_keeps_original_closed_zero_tie() {
    actual_sat(6.);
}
