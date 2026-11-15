// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Manual captures supply hints, not evidence of an actual optimizing LP call.
use super::super::test_fixture::Fixture;
use super::*;
use crate::single_hidden_hull::mip_cuts as root;
pub(in super::super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn lifted(x: f64) -> f64 {
    x.next_up().next_up()
}
pub(in super::super) fn pressure() -> Fixture {
    let (d, h) = (30, 128);
    let mut w = vec![0.; d * h];
    w[0] = 1.;
    for j in 1..62 {
        w[j * d + 2 + (j - 1) % 28] = 1.;
    }
    w[62 * d] = 1.;
    w[62 * d + 1] = 1.;
    w[63 * d] = 1.;
    w[63 * d + 1] = -1.;
    for j in 64..114 {
        w[j * d + 2 + (j - 64) % 28] = 1.;
    }
    let mut output = vec![0.; 2 * h];
    for value in &mut output[..62] {
        *value = lifted(1.);
    }
    for value in &mut output[62..114] {
        *value = lifted(-1.);
    }
    for value in &mut output[114..h] {
        *value = lifted(0.);
    }
    let mut bounds = vec![Bound::new(0., 0.); h];
    bounds[..114].fill(Bound::new(-1., 1.));
    bounds[62] = Bound::new(-2., 2.);
    bounds[63] = Bound::new(-2., 2.);
    Fixture {
        w: vec![w, output],
        b: vec![vec![0.; h], vec![0.25, 0.]],
        dims: vec![d, h, 2],
        input: vec![Bound::new(-1., 1.); d],
        bounds: vec![bounds],
    }
}
fn plan(p: &HullMarginProblem) -> Plan {
    let negative: Vec<_> = (0..p.hidden_dim).filter(|&j| p.c[j] < 0.).collect();
    let count = 16.min(negative.len());
    let mut chunks = vec![Vec::new(); count];
    for (i, j) in negative.into_iter().enumerate() {
        chunks[i % count].push(j);
    }
    let components = chunks
        .into_iter()
        .enumerate()
        .map(|(i, negative)| {
            let positive = if i == 0 { vec![0] } else { vec![] };
            Component {
                alpha: vec![0.; (1usize << negative.len()) * positive.len()],
                negative,
                positive_units: vec![2; positive.len()],
                positive,
                g: vec![0.; p.input_dim],
            }
        })
        .collect();
    Plan {
        components,
        leftover: (1..p.hidden_dim).filter(|&j| p.c[j] > 0.).collect(),
    }
}
pub(in super::super) fn manual(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    manual_capture(p, false)
}
pub(in super::super) fn manual_hint(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    manual_capture(p, true)
}
fn manual_capture(p: &HullMarginProblem, strict_hint: bool) -> HullRootMipCuts<'_> {
    let deadline = end();
    let mut attempt = primal_capture::Attempt::new(p, deadline);
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    attempt.admit(&roles);
    let lp = build_hull_lp_with_capture(p, &roles, &bounds, attempt.capture.as_mut()).unwrap();
    assert!(attempt.enable_same_call(lp.model.num_cols()));
    let mut raw = vec![0.; lp.model.num_cols()];
    if strict_hint {
        use crate::single_hidden_hull::Role;
        assert_eq!((p.input_dim, p.hidden_dim), (30, 128));
        let mut col = p.input_dim;
        for (j, role) in roles.iter().enumerate() {
            match role {
                Role::PosUnstable => {
                    raw[col] = if j == 0 { 0.5 } else { 0. };
                    col += 1;
                }
                Role::NegUnstable => {
                    let y = if j == 62 || j == 63 { 1. } else { 0. };
                    raw[col] = 1. - y / bounds[j].1;
                    col += 1;
                    for (i, &w) in p.w_row(j).iter().enumerate() {
                        let width = w.abs() * (p.upper[i] - p.lower[i]);
                        if width > 0. && width.is_finite() {
                            col += 1;
                        }
                    }
                }
                Role::Inert | Role::Dead | Role::Live => {}
            }
        }
        assert_eq!(col, lp.model.num_cols());
    }
    attempt.finish(Some(raw.into_boxed_slice()), &mut || true);
    let m = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let mut cuts = root::prepare(p, m, lower, deadline, &mut || true).unwrap();
    cuts.primal = attempt.take_account();
    cuts.primal_report = Some(Arc::clone(&attempt.report));
    cuts.group_allocation_mode = true;
    cuts.branch_tau_mode = true;
    let budget = Budget::new(&cuts, deadline, Duration::ZERO).unwrap();
    let plan = plan(p);
    let cost = plan.cost(p).unwrap();
    budget.reserve(cost).unwrap();
    budget.proof(cost, false).unwrap();
    let checked =
        certificate::check(&cuts, &plan, &budget, budget.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    let row = row(&cuts, &plan, checked, cost, &budget, &seal);
    component_rows::admit(&cuts, &plan, &budget).unwrap();
    assert!(row.lower < 0.);
    cuts.signed = Some(row);
    cuts
}
pub(in super::super) fn first(cuts: &HullRootMipCuts<'_>) {
    let row = cuts.signed.as_ref().unwrap();
    assert!(row
        .replay_inner(cuts, row.authority(), &mut || true, false)
        .is_some());
    let c = row.budget.state.lock().unwrap();
    assert_eq!((c.replay_attempts, c.source_replays), (1, 1));
}
pub(in super::super) fn hint(p: &HullMarginProblem, positive: f64, negative: f64) -> Point {
    let mut values = vec![0.; p.hidden_dim];
    values[0] = positive;
    values[62] = negative;
    values[63] = negative;
    Point {
        input: vec![0.; p.input_dim].into_boxed_slice(),
        values: values.into_boxed_slice(),
        positive: (0..62).collect::<Vec<_>>().into_boxed_slice(),
        negative: (62..114).collect::<Vec<_>>().into_boxed_slice(),
    }
}
