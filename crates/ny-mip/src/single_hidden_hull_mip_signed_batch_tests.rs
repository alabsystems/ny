// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fixed exact graph and ledger controls, not target/runtime claims.
use super::super::tests::{cuts, end, graph, rat};
use super::*;

pub(super) fn family(sizes: &[usize], bias: f64) -> (HullMarginProblem, Plan) {
    let d = 30;
    let h = sizes.iter().sum::<usize>() * 3;
    let mut w = vec![0.; d * h];
    let mut c = vec![0.; h];
    let mut components = Vec::new();
    let mut head = 0;
    let mut axis = 0;
    for (group, &n) in sizes.iter().enumerate() {
        let first = head;
        for j in 0..3 * n {
            let input = if group + 1 == sizes.len() && j % n + 1 == n {
                d - 1
            } else {
                axis + j % n
            };
            w[head * d + input] = 1.;
            c[head] = if j < 2 * n { 0.5 } else { -1. };
            head += 1;
        }
        components.push(Component {
            negative: (first + 2 * n..head).collect(),
            positive: (first..first + 2 * n).collect(),
            positive_units: vec![2; 2 * n],
            g: vec![0.; d],
            alpha: vec![0.5; (1 << n) * 2 * n],
        });
        axis += n;
    }
    let mut p = HullMarginProblem::new(
        w,
        vec![0.; h],
        d,
        &c,
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    // Exact authored dyadic graph: positive twins cancel every negative ReLU.
    // These controls do not assert an IEEE/encoder envelope or original G3.
    p.c = c;
    p.d = bias;
    (
        p,
        Plan {
            components,
            leftover: vec![],
        },
    )
}
pub(super) fn baseline(
    c: &HullRootMipCuts<'_>,
    p: &Plan,
    b: &Budget,
) -> (usize, f64, SignedMajorant) {
    let cost = p.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let value = certificate::check(c, p, b, b.search_end, &mut || true).unwrap();
    let lower = value.root_lower;
    let seal = Arc::new(signature(c).collect());
    (cost, lower, row(c, p, value, cost, b, &seal))
}
pub(super) fn leave(b: &Budget, keep: usize) {
    b.charge(b.remaining().unwrap().checked_sub(keep).unwrap())
        .unwrap();
}
pub(super) fn run(
    c: &HullRootMipCuts<'_>,
    plan: &mut Plan,
    b: &Budget,
    cost: usize,
    current: &mut f64,
    best: &mut Option<SignedMajorant>,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Accepted> {
    let seal = Arc::new(signature(c).collect());
    refine(
        c,
        plan,
        current,
        best,
        c.checked_lower - 10.,
        cost,
        b,
        &seal,
        poll,
    )
}
pub(super) fn sum_min(p: &HullMarginProblem, plan: &Plan) -> usize {
    plan.components
        .iter()
        .map(|c| search::work(p, c).unwrap().cost(1).unwrap())
        .sum()
}
#[test]
fn signed_batch_fair_complete_masks_reach_dense_and_cheap_components_before_one_checker() {
    let (p, plan) = family(&[8, 1], 0.25);
    let c = cuts(&p);
    let dense = search::work(&p, &plan.components[0]).unwrap();
    let cheap = search::work(&p, &plan.components[1]).unwrap();
    let old = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, _, _) = baseline(&c, &plan, &old);
    let keep = cost + dense.cost(2).unwrap() + cheap.cost(1).unwrap() - 1;
    leave(&old, keep);
    let _dense = search::optimize(
        &c,
        &plan.components[0],
        vec![0.; 30],
        &old,
        cost,
        &mut || true,
    )
    .unwrap();
    assert_eq!(old.state.lock().unwrap().completed_masks, 256);
    assert!(search::optimize(
        &c,
        &plan.components[1],
        vec![0.; 30],
        &old,
        cost,
        &mut || true
    )
    .is_none());

    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    let before = current;
    let mut best = Some(previous);
    let mut trial = plan.clone();
    leave(&b, keep);
    let accepted = run(
        &c,
        &mut trial,
        &b,
        cost,
        &mut current,
        &mut best,
        &mut || true,
    )
    .unwrap();
    assert_eq!(accepted.refined, vec![true, true]);
    assert!(plan
        .components
        .iter()
        .all(|c| c.g.iter().all(|v| *v == 0.) && c.alpha.iter().all(|v| *v == 0.5)));
    assert_eq!(b.state.lock().unwrap().completed_masks, 258);
    assert_eq!(b.state.lock().unwrap().checked_masks, 516);
    assert_eq!(b.state.lock().unwrap().checks, 2);
    assert!(current > before + 0.4 && current <= p.d);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
    for _ in 0..2 {
        assert_eq!(
            best.as_ref()
                .unwrap()
                .replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            current.to_bits()
        );
    }
    assert_eq!(c.proposal_lp_calls, 0);
    assert!(b.state.lock().unwrap().work <= MAX_WORK);
}
#[test]
fn signed_batch_homogeneous_expensive_components_are_eligible_under_actual_work_pressure() {
    let (p, mut plan) = family(&[8, 8], 0.25);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    let keep = overhead(&p, &plan).unwrap() + cost + sum_min(&p, &plan);
    leave(&b, keep);
    let mut best = Some(previous);
    let accepted = run(
        &c,
        &mut plan,
        &b,
        cost,
        &mut current,
        &mut best,
        &mut || true,
    )
    .unwrap();
    assert_eq!(accepted.refined, vec![true, true]);
    assert_eq!(b.state.lock().unwrap().completed_masks, 512);
    assert_eq!(b.state.lock().unwrap().checked_masks, 1024);
    assert_eq!(b.state.lock().unwrap().checks, 2);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
}
#[test]
fn signed_batch_affordable_legacy_schedule_keeps_original_current_and_zero_seed_path() {
    let p = super::super::tests::twins(0.25, true);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let mut plan = geometry::plan(&c, &b).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    let mut best = Some(previous);
    let before = witness(&plan);
    assert!(run(
        &c,
        &mut plan,
        &b,
        cost,
        &mut current,
        &mut best,
        &mut || true
    )
    .is_none());
    assert_eq!(witness(&plan), before);
    assert_eq!(b.state.lock().unwrap().completed_masks, 0);
    assert_eq!(b.state.lock().unwrap().checks, 1);
    let (row, _) = propose(&c, end(), Duration::ZERO);
    assert!(row.unwrap().lower > 0.249);
}
#[test]
fn signed_batch_positive_current_g_retains_a_pending_zero_seed_that_recovers_quarter_strength() {
    let mut p = super::super::tests::twins(0.25, false);
    p.c = vec![1., 1., -1., -1.];
    p.d = 0.25;
    let c = cuts(&p);
    let mut plan = Plan {
        components: vec![
            Component {
                negative: vec![2],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![1. / 18., 0.],
                alpha: vec![0.5; 2],
            },
            Component {
                negative: vec![3],
                positive: vec![1],
                positive_units: vec![2],
                g: vec![0.; 2],
                alpha: vec![0.5; 2],
            },
        ],
        leftover: vec![],
    };
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (cost, mut current, previous) = baseline(&c, &plan, &b);
    b.state.lock().unwrap().checks = MAX_CHECKS - 2;
    let mut best = Some(previous);
    let accepted = run(
        &c,
        &mut plan,
        &b,
        cost,
        &mut current,
        &mut best,
        &mut || true,
    )
    .unwrap();
    assert!(current > 0.124 && current < 0.126);
    assert_eq!(accepted.pending_zero, vec![true, false]);
    let zero =
        search::optimize(&c, &plan.components[0], vec![0.; 2], &b, cost, &mut || true).unwrap();
    plan.components[0] = zero;
    b.proof(cost, false).unwrap();
    let final_check = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    assert!(final_check.root_lower > 0.249 && final_check.root_lower <= 0.25);
    assert_eq!(b.state.lock().unwrap().checks, MAX_CHECKS);
    assert_eq!(b.state.lock().unwrap().reserved, 2 * cost);
}
#[test]
fn signed_batch_unsafe_and_tie_original_graphs_never_get_a_strict_positive_floor() {
    for bias in [-0.25, 0.] {
        let (p, mut plan) = family(&[2, 1], bias);
        let c = cuts(&p);
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let (cost, mut current, previous) = baseline(&c, &plan, &b);
        leave(&b, overhead(&p, &plan).unwrap() + cost + sum_min(&p, &plan));
        let mut best = Some(previous);
        assert!(run(
            &c,
            &mut plan,
            &b,
            cost,
            &mut current,
            &mut best,
            &mut || true
        )
        .is_some());
        assert!(current <= 0.);
        for value in [-1., 0., 1.] {
            let mut x = vec![value; 30];
            x[29] = -value;
            assert_eq!(graph(&p, &x), rat(bias));
            assert!(rat(current) <= graph(&p, &x));
        }
    }
}
