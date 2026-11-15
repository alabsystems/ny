// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact supplied-prefix separation and complete original-Source checks.
//! These controls do not claim automatic partition discovery.
use super::test_fixture::{baseline, end};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}

fn strict_source() -> (
    HullMarginProblem,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<Bound>>,
) {
    let (d, h) = (13, 13);
    let mut first = vec![0.; d * h];
    first[0] = 1.;
    for j in 1..=6 {
        first[j * d] = 1.;
        first[j * d + j] = 1. / 256.;
    }
    for j in 7..h {
        first[j * d + j] = 1.;
    }
    let mut output = vec![0.; 2 * h];
    for j in 0..h {
        let c = if j == 0 {
            6_f64
        } else if j <= 6 {
            -1_f64
        } else {
            -1_f64 / 128.
        };
        output[j] = c.next_up().next_up();
    }
    let weights = vec![first.clone(), output.clone()];
    let biases = vec![vec![0.; h], vec![0.25, 0.]];
    let bounds = vec![(0..h)
        .map(|j| {
            if (1..=6).contains(&j) {
                Bound::new(-257. / 256., 257. / 256.)
            } else {
                Bound::new(-1., 1.)
            }
        })
        .collect()];
    let p = HullMarginProblem::new(
        first,
        vec![0.; h],
        d,
        &output,
        &[0.25, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    assert_eq!(p.c[0], 6.);
    assert!(p.c[1..7].iter().all(|c| *c == -1.));
    assert!(p.c[7..].iter().all(|c| *c == -1. / 128.));
    (p, weights, biases, bounds)
}
fn strict_plan(p: &HullMarginProblem) -> Plan {
    Plan {
        components: [vec![1, 2, 7, 8, 9, 10], vec![3, 4, 5, 6, 11, 12]]
            .into_iter()
            .map(|negative| Component {
                negative,
                positive: vec![0],
                positive_units: vec![1],
                g: vec![0.; p.input_dim],
                alpha: vec![0.5; 64],
            })
            .collect(),
        leftover: vec![],
    }
}
fn checked_context(
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
    let c = prepare(p, m, lower, end(), &mut || true).unwrap();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    (c, plan, b, checked, cost)
}

#[test]
fn root_coupling_six_selected_phases_strictly_separate_the_original_scalar_phase_q3_prefix() {
    let (p, w, biases, bounds) = strict_source();
    let (mut cuts, mut plan, mut budget, mut old, mut cost) = checked_context(&p, strict_plan(&p));
    let original_lower = old.root_lower;
    assert!(original_lower < 0.);
    // Every permitted scalar/phase component projection is an original graph
    // mixture. Only the weighted aggregate/phase projection is asserted here.
    let a = q(0.75);
    let z = q(1.25);
    let positive = q(0.25);
    for m in [0_f64, 3., 6.] {
        for (coefficient, wanted, t) in [
            (
                m - 2.,
                q(m) * &positive - &a,
                match m as usize {
                    0 => q(0.75),
                    3 => q(0.),
                    _ => q(0.375),
                },
            ),
            (
                2. - m,
                q(6. - m) * &positive - &z,
                match m as usize {
                    0 => q(0.25),
                    3 => q(1.),
                    _ => q(0.625),
                },
            ),
        ] {
            assert_eq!(q(coefficient) * &t / q(2.), wanted);
            assert!(t >= Q::zero() && t <= q(1.));
            // Equal +/-t input atoms have mean0, meanReLU=t/2, phase1/2.
            assert_eq!((&t - &t) / q(2.), Q::zero());
        }
    }
    // ALL <=3 selected principal subsets leave at least3 principal tails.
    // A global affine upper at the origin must be >=1/2, independently of G.
    for mask in 0usize..(1usize << 12) {
        if mask.count_ones() > 3 {
            continue;
        }
        let omitted = (1..=6).filter(|j| mask & (1usize << (j - 1)) == 0).count();
        assert!(omitted >= 3);
        assert!(-q(omitted as f64) / q(2.) <= q(-1.5));
        assert!(q(-0.5) >= -q(omitted as f64) / q(2.));
    }
    let mut encoder = crate::encode_feedforward(
        &w,
        &biases,
        &[13, 13, 2],
        &[Bound::new(-1., 1.); 13],
        &bounds,
    )
    .unwrap();
    encoder.constrain_output_leq_row(0, 1).unwrap();
    let original = encoder.into_parts();
    let unsafe_row = Row(original.problem.num_rows() - 1);
    let encoding = HullMipEncoding {
        weights: &w,
        biases: &biases,
        dimensions: &[13, 13, 2],
        hidden_bounds: &bounds,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row,
    };
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let mut fake = vec![Q::zero(); original.num_cols];
    for i in 7..13 {
        fake[original.input_vars[i].0] = q(-1.);
    }
    let mut values = vec![Q::zero(); 13];
    values[0] = q(0.25);
    values[1] = q(0.375);
    values[2] = q(0.375);
    for x in &mut values[3..7] {
        *x = q(0.3125);
    }
    for j in 0..13 {
        fake[13 + j] = if j >= 7 { q(-1.) } else { Q::zero() };
        fake[heads[j].0] = values[j].clone();
        fake[original.binary_vars[j].0] = if j >= 7 { Q::zero() } else { q(0.5) };
    }
    fake[original.output_vars[0].0] = q(0.25) + (0..13).map(|j| q(w[1][j]) * &values[j]).sum::<Q>();
    assert!(fake[original.output_vars[0].0] < Q::zero());
    for (j, c) in original.problem.cols().iter().enumerate() {
        assert!(!c.lb.is_finite() || fake[j] >= q(c.lb));
        assert!(!c.ub.is_finite() || fake[j] <= q(c.ub));
    }
    for r in original.problem.rows() {
        let lhs: Q = r.coeffs.iter().map(|(j, a)| q(*a) * &fake[*j]).sum();
        assert!(!r.lb.is_finite() || lhs >= q(r.lb));
        assert!(!r.ub.is_finite() || lhs <= q(r.ub));
    }
    let floor = cuts.checked_lower;
    assert!(run(
        &cuts,
        &mut plan,
        &mut old,
        &mut cost,
        floor,
        &mut budget,
        &mut || true
    ));
    let report = budget.state.lock().unwrap().root_coupling.clone();
    assert!(report.selected && report.completed_supports == 1);
    assert_eq!((report.old_masks, report.new_masks), (128, 76));
    assert_eq!((report.selected_negatives, report.tail_negatives), (6, 6));
    assert!(old.root_lower > 23. / 128. - 1e-10 && old.root_lower <= 23. / 128. + 1e-12);
    assert!(old.root_lower > original_lower);
    component_rows::admit(&cuts, &plan, &budget).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, &plan, old, cost, &budget, &seal));
    let mut parts = original;
    let col = parts.binary_vars.remove(0);
    parts.binary_widths.remove(0);
    assert!(parts.problem.relax_binary_integrality(&[col]));
    let before = parts.problem.num_rows();
    let appended = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(appended.proposal_lp_calls, 0);
    assert!(parts.problem.rows()[before..].iter().any(|r| {
        let lhs: Q = r.coeffs.iter().map(|(j, a)| q(*a) * &fake[*j]).sum();
        (r.lb.is_finite() && lhs < q(r.lb)) || (r.ub.is_finite() && lhs > q(r.ub))
    }));
    let proof = cuts.signed_root_proof(end()).unwrap();
    assert!(proof.verified() && proof.lower() > 0.);
    assert_eq!(cuts.signed.as_ref().unwrap().stats().source_replays, 2);
}

#[test]
fn root_coupling_half_units_above_sixteen_positives_use_a_ranked_tail_recipient_without_mass_loss()
{
    let h = 21;
    let mut output = vec![0.; 2 * h];
    for j in 0..h {
        let c = if j < 17 { (j + 1) as f64 } else { -100_f64 };
        output[j] = c.next_up().next_up();
    }
    let p = HullMarginProblem::new(
        vec![1.; h],
        vec![0.; h],
        1,
        &output,
        &[-100., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    let pos = (0..16).collect::<Vec<_>>();
    let make = |j, positive: Vec<usize>| Component {
        negative: vec![j],
        positive_units: vec![1; positive.len()],
        alpha: vec![0.5; 2 * positive.len()],
        positive,
        g: vec![0.],
    };
    let plan = Plan {
        components: vec![
            make(17, pos.clone()),
            make(18, vec![16]),
            make(19, pos),
            make(20, vec![16]),
        ],
        leftover: vec![],
    };
    let (c, plan, mut b, _, cost) = checked_context(&p, plan);
    let (t, _) = admit(&c, &plan, cost, &mut b).unwrap();
    t.begin(&c, &b).unwrap();
    t.claim_selection(&c, &b, &plan, cost).unwrap();
    let check = cost + resources::component_cost(1, 6, 16, 16).unwrap() + 10000;
    let search =
        resources::search_cost(1, 1, 16, 6).unwrap() + resources::search_cost(1, 1, 1, 1).unwrap();
    let choice = resources::Choice {
        left: 0,
        right: 1,
        selected: 1,
        positive: 16,
        recipient: 1,
        active: 6,
        check,
        search,
        total: check + search,
        score: 1.,
    };
    t.bind_union_choice(&c, &b, &choice, &[choice.left, choice.right])
        .unwrap();
    t.pay(&c, &b, choice.total).unwrap();
    let built = geometry::build(&c, &plan, &choice, &b, &t, &mut || true).unwrap();
    built.plan.valid(&c).unwrap();
    let selected = &built.plan.components[built.selected];
    assert_eq!(selected.positive.len(), 16);
    assert!(!selected.positive.contains(&0));
    assert_eq!(
        built.plan.components[built.recipient.unwrap()].positive,
        vec![0]
    );
    for j in 0..17 {
        let total: usize = built
            .plan
            .components
            .iter()
            .map(|c| {
                c.positive
                    .iter()
                    .position(|x| *x == j)
                    .map_or(0, |i| usize::from(c.positive_units[i]))
            })
            .sum();
        assert_eq!(total, 2);
    }
    let mut corrupt = built.plan.clone();
    corrupt.leftover.push(0);
    assert!(corrupt.valid(&c).is_none());
    assert!(certificate::check(&c, &corrupt, &b, t.end, &mut || true).is_none());
}

#[test]
fn root_coupling_folded_direction_includes_leftover_positive_affine_and_original_roundoff() {
    let mut p = test_fixture::source(-0.25);
    p.hidden_dim = 4;
    p.w1.push(-2.);
    p.b1.push(0.5);
    p.c.push(0.25);
    let mut m = HullMultipliers::zeros(&p);
    m.alpha[3] = 0.75;
    let lower = certify_hull_lower_bound(&p, &m).unwrap();
    let c = prepare(&p, m, lower, end(), &mut || true).unwrap();
    let mut plan = test_fixture::plan(&p);
    plan.leftover.push(3);
    plan.valid(&c).unwrap();
    let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let (t, _) = admit(&c, &plan, cost, &mut b).unwrap();
    t.begin(&c, &b).unwrap();
    let choice = resources::choose(&c, &plan, &b, &t, cost, &mut || true).unwrap();
    t.pay(&c, &b, choice.total).unwrap();
    let built = geometry::build(&c, &plan, &choice, &b, &t, &mut || true).unwrap();
    assert_eq!(built.plan.components[built.selected].g, vec![0.375]);
    let checked = certificate::check(&c, &built.plan, &b, t.end, &mut || true).unwrap();
    for x in [-1_f64, -0.5, 0., 0.25, 0.5, 1.] {
        let value = p.d
            + (0..4)
                .map(|j| p.c[j] * (p.w_row(j)[0] * x + p.b1[j]).max(0.))
                .sum::<f64>();
        assert!(checked.root_lower <= value + 1e-13);
        assert!(checked.constant + checked.g[0] * x <= value - p.d + 1e-13);
    }
}

#[test]
fn root_coupling_existing_search_quotes_cover_complete_masks_and_finite_overflow_refuses() {
    assert_eq!(search::coordinate_work(30), Some(4632));
    assert_eq!(resources::search_cost(30, 6, 16, 3), Some(1_961_216));
    assert_eq!(resources::search_cost(30, 6, 16, 6), Some(3_739_904));
    assert_eq!(resources::search_cost(30, 1, 16, 16), Some(300_912));
    assert!(resources::search_cost(usize::MAX, 6, 16, 3).is_none());
    let p = test_fixture::source(-0.25);
    let (c, plan, mut b, _, cost) = baseline(&p);
    let (t, _) = admit(&c, &plan, cost, &mut b).unwrap();
    t.begin(&c, &b).unwrap();
    let choice = resources::choose(&c, &plan, &b, &t, cost, &mut || true).unwrap();
    assert!(choice.check >= cost - resources::component_cost(1, 1, 1, 1).unwrap() * 2);
    assert!(choice.total <= t.remaining(&c, &b).unwrap());
    t.pay(&c, &b, choice.total).unwrap();
    let mut built = geometry::build(&c, &plan, &choice, &b, &t, &mut || true).unwrap();
    built.plan.components[built.selected].g[0] = f64::MAX;
    built.plan.components[built.selected].alpha.fill(f64::NAN);
    assert!(certificate::check(&c, &built.plan, &b, t.end, &mut || true).is_none());
}
