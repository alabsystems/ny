// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fixed-family controls, not a whole-NY separation or runtime claim.
use super::*;
use crate::single_hidden_hull::mip_cuts::signed::tests::{cuts as twin_cuts, end, twins};
use num_rational::BigRational as Q;
use num_traits::Zero;

fn family(bias: f64, d: usize) -> HullMarginProblem {
    let mut w = vec![0.; 4 * d];
    for (j, (x, y)) in [(1., 1.), (1., -1.), (-1., 1.), (-1., -1.)]
        .into_iter()
        .enumerate()
    {
        w[j * d] = x;
        w[j * d + d - 1] = y;
    }
    let mut p = HullMarginProblem::new(
        w,
        vec![0.; 4],
        d,
        &[-1.; 4],
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    // Authored exact dyadic native coefficients; no encoder/Source-E1 claim.
    p.c.fill(-1.);
    p.d = bias;
    p
}
fn setup(p: &HullMarginProblem, zero: bool) -> (HullRootMipCuts<'_>, Plan) {
    let mut m = HullMultipliers::zeros(p);
    for j in 0..4 {
        m.g[j * p.input_dim] = if j < 2 { 1. } else { -1. };
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let c = prepare(p, m, lower, end(), &mut || true).unwrap();
    let mut a = vec![0.; p.input_dim];
    let mut z = a.clone();
    if !zero {
        a[0] = -2.;
        z[0] = 2.;
    }
    let plan = Plan {
        components: vec![
            Component {
                negative: vec![0, 1],
                positive: vec![],
                positive_units: vec![],
                g: a,
                alpha: vec![],
            },
            Component {
                negative: vec![2, 3],
                positive: vec![],
                positive_units: vec![],
                g: z,
                alpha: vec![],
            },
        ],
        leftover: vec![],
    };
    (c, plan)
}
fn baseline(c: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) -> (usize, certificate::Checked) {
    let cost = plan.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(c, plan, b, b.search_end, &mut || true).unwrap();
    (cost, checked)
}
fn graph(p: &HullMarginProblem, x: &[f64]) -> Q {
    Q::from_float(p.d).unwrap()
        + (0..p.hidden_dim)
            .map(|j| {
                let z = Q::from_float(p.b1[j]).unwrap()
                    + (0..p.input_dim)
                        .map(|i| {
                            Q::from_float(p.w_row(j)[i]).unwrap() * Q::from_float(x[i]).unwrap()
                        })
                        .sum::<Q>();
                Q::from_float(p.c[j]).unwrap() * z.max(Q::zero())
            })
            .sum::<Q>()
}
#[test]
fn signed_direction_neutral_whole_sweeps_strictly_improve_both_zero_and_attaining_seeds() {
    for zero in [false, true] {
        let p = family(2.25, 2);
        let (c, mut plan) = setup(&p, zero);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, initial) = baseline(&c, &plan, &b);
        assert!(initial.root_lower <= -1.749);
        let mut current = initial.root_lower;
        let mut best = None;
        let seal = Arc::new(signature(&c).collect());
        refine(
            &c,
            &mut plan,
            &mut current,
            &mut best,
            c.checked_lower,
            cost,
            &b,
            &seal,
            &mut || true,
        );
        assert!(current > 0.249 && current <= 0.25);
        assert!((plan.components[0].g[0] + 1.).abs() < 1e-12);
        assert!((plan.components[1].g[0] - 1.).abs() < 1e-12);
        let row = best.unwrap();
        let d = row.stats().direction;
        assert_eq!(d.completed_sweeps, 2);
        assert_eq!(d.checked_candidates, 1);
        assert_eq!(d.accepted_candidates, 1);
        assert!(d.coordinates > 0 && d.cached_masks == 8);
        assert_eq!(c.proposal_lp_calls, 0);
        assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
        for _ in 0..2 {
            assert!(
                row.replay_inner(&c, end(), &mut || true, false)
                    .unwrap()
                    .root_lower
                    > 0.249
            );
        }
    }
}
#[test]
fn signed_direction_intrinsic_breakpoint_requires_a_neutral_move_before_strict_gain() {
    let f = |a: f64, b: f64| 2.25 + a.min(-2. - a) + (-2. + b).min(-b) - (a + b).abs();
    assert_eq!(f(-2., 2.), -1.75);
    assert_eq!(f(-1., 2.), -1.75);
    assert_eq!(f(-1., 1.), 0.25);
    assert_eq!(f(0., 0.), -1.75);
    assert_eq!(f(-1., 0.), -1.75);
    // The four-head merger already closes: only this partition is compared.
    let p = family(2.25, 2);
    for x in [[1., 0.], [-1., 1.], [0., -1.]] {
        assert!(graph(&p, &x) >= Q::from_float(0.25).unwrap());
    }
}
#[test]
fn signed_direction_unsafe_and_tie_original_points_cannot_become_positive() {
    for bias in [1.75, 2.] {
        let p = family(bias, 2);
        let (c, mut plan) = setup(&p, false);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, initial) = baseline(&c, &plan, &b);
        let mut lower = initial.root_lower;
        let mut best = None;
        let seal = Arc::new(signature(&c).collect());
        refine(
            &c,
            &mut plan,
            &mut lower,
            &mut best,
            c.checked_lower,
            cost,
            &b,
            &seal,
            &mut || true,
        );
        let actual = graph(&p, &[1., 0.]);
        assert!(actual <= Q::zero());
        assert!(Q::from_float(lower).unwrap() <= actual && lower <= 0.);
        assert!(best.is_some());
    }
}
#[test]
fn signed_direction_default_propose_exercises_safe_unsafe_tie_and_real_report() {
    for bias in [0.25, -0.25, 0.] {
        let p = twins(bias, false);
        let c = twin_cuts(&p);
        let (row, report) = propose(&c, end(), Duration::ZERO);
        assert_eq!(report.direction.attempts, 1);
        assert_eq!(report.direction.completed_sweeps, 2);
        assert!(report.direction.work_units > 0 && report.direction.cached_masks > 0);
        let row = row.unwrap();
        if bias > 0. {
            assert!(row.lower > 0.249 && row.lower <= 0.25);
        } else {
            assert!(row.lower <= bias && row.lower <= 0.);
        }
    }
}
#[test]
fn signed_direction_partial_cache_or_sweep_refusal_does_not_publish_a_nominal_plan() {
    for limit in [1, 8] {
        let p = family(2.25, 2);
        let (c, plan) = setup(&p, false);
        let before = witness(&plan);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, _) = baseline(&c, &plan, &b);
        let mut poll = || b.state.lock().unwrap().direction.cached_masks < limit;
        assert!(nominal(&c, &plan, &b, cost, &mut poll).is_none());
        assert_eq!(witness(&plan), before);
        assert_eq!(b.state.lock().unwrap().direction.completed_sweeps, 0);
        assert_eq!(b.state.lock().unwrap().direction.checked_candidates, 0);
        assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    }
}
#[test]
fn signed_direction_later_full_check_refusal_retains_completed_row_and_two_replays() {
    let p = family(1.75, 2);
    let (c, mut plan) = setup(&p, false);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, initial) = baseline(&c, &plan, &b);
    let lower = initial.root_lower;
    let seal = Arc::new(signature(&c).collect());
    let mut best = Some(row(&c, &plan, initial, cost, &b, &seal));
    let mut current = lower;
    let before = witness(&plan);
    let mut poll = || b.state.lock().unwrap().direction.completed_sweeps < 2;
    refine(
        &c,
        &mut plan,
        &mut current,
        &mut best,
        c.checked_lower,
        cost,
        &b,
        &seal,
        &mut poll,
    );
    assert_eq!(witness(&plan), before);
    assert_eq!(current.to_bits(), lower.to_bits());
    assert_eq!(b.state.lock().unwrap().direction.checked_candidates, 0);
    for _ in 0..2 {
        assert_eq!(
            best.as_ref()
                .unwrap()
                .replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            lower.to_bits()
        );
    }
}
#[test]
fn signed_direction_optional_work_and_payload_refuse_without_spending_publication_reserve() {
    let p = family(1.75, 2);
    let (c, plan) = setup(&p, false);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, proof) = baseline(&c, &plan, &b);
    let seal = Arc::new(signature(&c).collect());
    let completed = row(&c, &plan, proof, cost, &b, &seal);
    b.charge(b.remaining().unwrap()).unwrap();
    assert!(nominal(&c, &plan, &b, cost, &mut || true).is_none());
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    for _ in 0..2 {
        assert!(completed
            .replay_inner(&c, end(), &mut || true, false)
            .is_some());
    }
    let mut tight = Budget::new(&c, end(), Duration::ZERO).unwrap();
    tight.admitted_bytes = MAX_BYTES;
    assert!(nominal(&c, &plan, &tight, cost, &mut || true).is_none());
    assert_eq!(tight.state.lock().unwrap().direction.stop_reason, "payload");
    assert_eq!(tight.state.lock().unwrap().direction.cache_bytes, 0);
}
#[test]
fn signed_direction_fixed_box_and_nonfinite_or_foreign_inputs_decline() {
    let mut p = family(2.25, 2);
    p.lower = vec![0.; 2];
    p.upper = vec![0.; 2];
    let (c, plan) = setup(&p, false);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    assert!(nominal(&c, &plan, &b, plan.cost(&p).unwrap(), &mut || true).is_none());
    assert_eq!(b.state.lock().unwrap().direction.stop_reason, "fixed_box");
    let p = family(2.25, 2);
    let (c, mut plan) = setup(&p, false);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    plan.components[0].g[1] = f64::NAN;
    assert!(nominal(&c, &plan, &b, plan.cost(&p).unwrap(), &mut || true).is_none());
    let (c, mut huge) = setup(&p, false);
    huge.components[0].g.fill(f64::MAX);
    let before = witness(&huge);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    assert!(nominal(&c, &huge, &b, huge.cost(&p).unwrap(), &mut || true).is_none());
    assert_eq!(witness(&huge), before); // An overflowing cache never commits any tuple.
    let other = p.clone();
    let (foreign, plan) = setup(&other, false);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    assert!(
        nominal(&foreign, &plan, &b, plan.cost(&other).unwrap(), &mut || {
            true
        })
        .is_none()
    );
}
#[test]
fn signed_direction_last_coordinate_asymmetric_box_and_rounding_remain_source_priced() {
    let mut p = family(2.25, 30);
    p.lower[29] = -0.75;
    p.upper[29] = 0.5;
    let (c, mut plan) = setup(&p, false);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, initial) = baseline(&c, &plan, &b);
    let mut current = initial.root_lower;
    let mut best = None;
    let seal = Arc::new(signature(&c).collect());
    refine(
        &c,
        &mut plan,
        &mut current,
        &mut best,
        c.checked_lower,
        cost,
        &b,
        &seal,
        &mut || true,
    );
    assert!(current.is_finite());
    for (x, y) in [(-1., -0.75), (-1., 0.5), (1., -0.75), (1., 0.5)] {
        let mut input = vec![0.; 30];
        input[0] = x;
        input[29] = y;
        assert!(Q::from_float(current).unwrap() <= graph(&p, &input));
    }
    assert_eq!(c.proposal_lp_calls, 0);
}
#[test]
fn signed_direction_original_expiry_and_exhausted_check_slots_never_renew() {
    let p = family(1.75, 2);
    let (c, plan) = setup(&p, false);
    let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    b.search_end = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(nominal(&c, &plan, &b, plan.cost(&p).unwrap(), &mut || true).is_none());
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    b.state.lock().unwrap().checks = MAX_CHECKS;
    assert!(nominal(&c, &plan, &b, plan.cost(&p).unwrap(), &mut || true).is_none());
    assert_eq!(b.state.lock().unwrap().direction.work_units, 0);
    assert_eq!(b.state.lock().unwrap().direction.stop_reason, "checks");
}
