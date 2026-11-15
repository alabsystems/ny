// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied ordinary plans for engineering controls, not default discovery.
use super::*;
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn source(bias: f64) -> HullMarginProblem {
    let out = vec![
        4_f64.next_up().next_up(),
        (-1_f64).next_up().next_up(),
        (-3_f64).next_up().next_up(),
        0.,
        0.,
        0.,
    ];
    let p = HullMarginProblem::new(
        vec![1.; 3],
        vec![0.; 3],
        1,
        &out,
        &[bias, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    assert_eq!(p.c, vec![4., -1., -3.]);
    p
}
pub(super) fn plan(p: &HullMarginProblem) -> Plan {
    Plan {
        components: [1, 2]
            .into_iter()
            .map(|j| Component {
                negative: vec![j],
                positive: vec![0],
                positive_units: vec![1],
                g: vec![0.; p.input_dim],
                alpha: vec![0.5; 2],
            })
            .collect(),
        leftover: vec![],
    }
}
pub(super) fn baseline(
    p: &HullMarginProblem,
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
    let plan = plan(p);
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    (cuts, plan, b, checked, cost)
}
