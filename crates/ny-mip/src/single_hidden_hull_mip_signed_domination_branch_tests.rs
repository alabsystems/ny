// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exclusive-tail engineering controls. Manual hints do not claim LP discovery.
use super::test_fixture::{end, first, hint, manual, manual_hint, pressure};
use super::*;
fn quote(
    p: &HullMarginProblem,
    c: usize,
    capture: usize,
    positive: usize,
    negative: usize,
    q: usize,
) -> Option<Quote> {
    let base = dense::quote_with_aliases(p, c, capture, positive, negative, q)?;
    Planning::new(p)?.quote_base(p, c, capture, base)
}
#[test]
fn branch_max_tail_quote_funds_adjacent_actual_and_singleton_shapes_without_refund() {
    let p = pressure().source();
    let a = quote(&p, 16, 148864, 62, 52, 22).unwrap();
    assert_eq!(
        (a.total_work(), a.incremental_work(), a.base.positive_limit),
        (661824, 512960, 22)
    );
    assert_eq!((a.single, a.branch, a.optimizer), (186336, 183936, 84160));
    assert_eq!(a.bytes(), 154048);
    assert_eq!(
        a.incremental,
        a.base.incremental_work - a.removed_old_tail + a.single.max(a.branch)
    );
    assert!(quote(&p, 16, 148864, 62, 52, 23).is_none());
    let a = quote(&p, 68, 153664, 60, 54, 7).unwrap();
    assert_eq!((a.total, a.single, a.branch), (660240, 121344, 153216));
    assert!(quote(&p, 68, 153664, 60, 54, 8).is_none());
    assert!(quote(&p, 129, 148864, 62, 52, 1).is_none());
    let other = p.clone();
    assert!(!a.matches(&other, 68));
    assert!(!a.matches(&p, 16));
}
#[test]
fn branch_seed_route_is_once_only_and_known_impossible_pair_keeps_gradient_capability() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    assert_eq!(
        resources::select_branch_route(row, &t, false, false),
        Some(true)
    );
    assert_eq!(resources::select_branch_route(row, &t, false, true), None);
    assert!(!resources::begin_branch_tail(row, &t));
    assert!(resources::branch_support_attempt(row, &t, t.end).is_none());
    assert!(resources::support_attempt(row, &t, t.end).is_some());
    assert!(resources::support_attempt(row, &t, t.end).is_none());
    let r = row.stats().component_rows.domination;
    assert_eq!(r.exclusive_seed_useful, Some(false));
    assert!(!r.gradient_skipped);
    assert_eq!(
        (r.advice_attempts, r.support_attempts, r.checker_slots),
        (1, 1, 1)
    );
}
#[test]
fn branch_nonpositive_seed_consumes_pair_tail_without_gradient_or_tail_restart() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    let point = hint(&p, 0.5, 1.);
    let a = dense::exclusive_advice(
        row,
        &cuts,
        &point,
        t.dense_quote().unwrap(),
        &t,
        t.end,
        &mut || true,
    )
    .unwrap();
    assert!(!a.useful);
    assert_eq!(a.partner, Some(63));
    let proposal = proposal(row, &cuts, &point, &t, &a, t.end, &mut || true).unwrap();
    assert_eq!(proposal.affine.negative, [62, 63]);
    assert!(group::kernel::branch::check(row, &cuts, &t, proposal, t.end, &mut || true).is_some());
    assert!(self::proposal(row, &cuts, &point, &t, &a, t.end, &mut || true).is_none());
    assert!(resources::support_attempt(row, &t, t.end).is_none());
    let r = row.stats().component_rows.domination;
    assert!(r.gradient_skipped);
    assert!(!r.priced_step_attempted);
    assert_eq!(
        (
            r.branch_tau_attempts,
            r.branch_tau_completed,
            r.branch_tau_checked_masks
        ),
        (1, 1, 4)
    );
}
#[test]
fn branch_pair_tail_cancellation_is_sticky_spent_and_cannot_switch_to_gradient() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    resources::select_branch_route(row, &t, false, true).unwrap();
    let point = hint(&p, 0.5, 1.);
    let a = dense::Advice {
        single: kernel::Proposal {
            head: 62,
            positive: vec![0],
            lambda: vec![1.],
            v: vec![0.; 30],
        },
        useful: false,
        partner: Some(63),
    };
    let before = row.budget.state.lock().unwrap().work;
    let mut polls = 0;
    assert!(proposal(row, &cuts, &point, &t, &a, t.end, &mut || {
        polls += 1;
        polls < 9
    })
    .is_none());
    assert_eq!(polls, 9);
    assert!(proposal(row, &cuts, &point, &t, &a, t.end, &mut || true).is_none());
    assert!(resources::select_branch_route(row, &t, true, true).is_none());
    assert!(resources::support_attempt(row, &t, t.end).is_none());
    let c = row.budget.state.lock().unwrap();
    assert_eq!(c.work, before);
    assert_eq!(
        (
            c.component_rows.domination.branch_tau_attempts,
            c.component_rows.domination.completed_supports
        ),
        (1, 0)
    );
    assert_eq!((c.replay_attempts, c.source_replays), (1, 1));
    assert!(c.reserved > 0);
}
#[test]
fn branch_synthetic_capture_runs_actual_publication_pair_orchestration_once() {
    let f = pressure();
    let p = f.source();
    let cuts = manual_hint(&p);
    let mut parts = f.parts();
    let before = parts.problem.rows().len();
    let r = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let d = &r.signed.as_ref().unwrap().component_rows.domination;
    assert!(d.branch_tau_mode && d.gradient_skipped);
    assert_eq!(
        (
            d.group_nominal_masks,
            d.branch_tau_checked_masks,
            d.appended_rows
        ),
        (4, 4, 1)
    );
    assert!(parts.problem.rows().len() > before);
    let row = cuts.signed.as_ref().unwrap();
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_some());
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_none());
    let d = row.stats().component_rows.domination;
    assert_eq!(
        (
            d.branch_tau_attempts,
            d.branch_tau_completed,
            d.support_attempts,
            d.appended_rows
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(row.stats().source_replays, 2);
}
#[test]
fn branch_gradient_only_improvement_is_an_explicit_exclusive_schedule_tradeoff() {
    let mut f = pressure();
    let d = 30;
    f.w[0][d..2 * d].fill(0.);
    f.w[0][d] = 1.;
    f.w[0][d + 1] = 1.;
    for j in [62, 63] {
        f.w[0][j * d..(j + 1) * d].fill(0.);
        f.w[0][j * d] = 2.;
        f.w[0][j * d + 1] = 1.;
    }
    f.bounds[0][1] = Bound::new(-2., 2.);
    f.bounds[0][62] = Bound::new(-3., 3.);
    f.bounds[0][63] = Bound::new(-3., 3.);
    let p = f.source();
    let make_point = || {
        let mut point = hint(&p, 0., 0.75);
        point.input[0] = -0.5;
        point.input[1] = 1.;
        point.values[1] = 0.5;
        point.values[63] = 0.5;
        point
    };
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    let a = dense::exclusive_advice(
        row,
        &cuts,
        &make_point(),
        t.dense_quote().unwrap(),
        &t,
        t.end,
        &mut || true,
    )
    .unwrap();
    assert!(!a.useful);
    assert!(row.stats().component_rows.domination.gradient_skipped);
    assert_eq!(a.single.positive, vec![0, 1]);
    assert_eq!(a.single.lambda, vec![2., 0.5]);
    let baseline = manual(&p);
    first(&baseline);
    let row = baseline.signed.as_ref().unwrap();
    let account = baseline.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_point(row, &baseline, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    let a = dense::advice(
        row,
        &baseline,
        &make_point(),
        t.dense_quote().unwrap(),
        t.end,
        &mut || true,
    )
    .unwrap();
    assert!(a.useful);
    let r = row.stats().component_rows.domination;
    assert!(r.priced_step_completed && r.priced_step_accepted);
}

#[test]
fn branch_quote_rejects_forged_removed_tail_and_total_without_debit() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let before = row.stats();
    let base = quote(&p, 16, 0, 62, 52, 22).unwrap();
    let capture = resources::MAX_DOMINATION_WORK
        .checked_sub(base.incremental_work())
        .unwrap();
    let exact = quote(&p, 16, capture, 62, 52, 22).unwrap();
    assert_eq!(
        exact.checked_total(&p, 16, capture),
        Some(resources::MAX_DOMINATION_WORK)
    );
    assert!(quote(&p, 16, capture.checked_add(1).unwrap(), 62, 52, 22).is_none());
    let other = p.clone();
    for field in 0..9 {
        let mut forged = quote(&p, 16, capture, 62, 52, 22).unwrap();
        match field {
            0 => forged.total -= 1,
            1 => forged.removed_old_tail += 1,
            2 => forged.removed_old_tail = usize::MAX,
            3 => forged.base.total_work += 1,
            4 => forged.single += 1,
            5 => forged.branch += 1,
            6 => forged.payload += 1,
            7 => forged.source = std::ptr::from_ref(&other) as usize,
            8 => forged.components += 1,
            _ => unreachable!(),
        }
        assert_eq!(forged.checked_total(&p, 16, capture), None);
        assert_eq!(row.stats().work_units, before.work_units);
        assert_eq!(row.stats().admitted_bytes, before.admitted_bytes);
    }
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    assert!(resources::admit_branch_point(row, &cuts, end(), Some(&account)).is_some());
}
