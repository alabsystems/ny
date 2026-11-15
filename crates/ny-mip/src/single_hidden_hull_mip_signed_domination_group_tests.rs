// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact marginal-hull separation and one paid single-or-pair capability.
use super::super::test_fixture::{lp, q, rows_hold};
use super::test_fixture::{
    append_pure, end, first, graph, hint, manual, manual_hint, pair, pressure, toy,
};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
// Supply the same requested-limit metadata to the production quote over a base.
fn quote(
    p: &HullMarginProblem,
    components: usize,
    capture: usize,
    positives: usize,
    negatives: usize,
    requested: usize,
) -> Option<Quote> {
    if negatives < 2 {
        return None;
    }
    let q = requested.min(MAX_TERMS.checked_sub(p.input_dim)?.checked_sub(K)?);
    let base = dense::quote_with_aliases(p, components, capture, positives, negatives, q)?;
    super::quote(p, components, capture, base)
}
fn affine(p: &HullMarginProblem, j: usize, x: &[Q]) -> Q {
    q(p.b1[j]) + p.w_row(j).iter().zip(x).map(|(&a, x)| q(a) * x).sum::<Q>()
}
fn oracle(p: &HullMarginProblem, a: &kernel::Proposal) -> Q {
    (0..1usize << p.input_dim)
        .map(|mask| {
            let x: Vec<_> = (0..p.input_dim)
                .map(|i| {
                    q(if mask & (1 << i) == 0 {
                        p.lower[i]
                    } else {
                        p.upper[i]
                    })
                })
                .collect();
            let b = a
                .positive
                .iter()
                .zip(&a.lambda)
                .map(|(&j, &a)| q(a) * affine(p, j, &x))
                .sum::<Q>();
            a.negative
                .iter()
                .map(|&j| (q(-p.c[j]) * affine(p, j, &x) - &b / q(2.)).max(Q::zero()))
                .sum::<Q>()
                - a.v.iter().zip(&x).map(|(&v, x)| q(v) * x).sum::<Q>()
        })
        .max()
        .unwrap()
}
#[test]
fn group_joint_price_strictly_separates_every_single_target_marginal_hull() {
    let f = toy();
    let p = f.source();
    assert_eq!(p.c, vec![1., -1., -1.]);
    let parts = f.parts();
    let heads = binding::head_columns(&f.encoding(), &f.original()).unwrap();
    for (head, plus) in [(1, [1., 1.]), (2, [1., -1.])] {
        let plus: Vec<_> = plus.into_iter().map(q).collect();
        let minus: Vec<_> = plus.iter().map(|x| -x).collect();
        let a = graph(&f, &parts, &plus, false);
        let b = graph(&f, &parts, &minus, true);
        rows_hold(&parts.problem, &a, Some(f.encoding().unsafe_row.0));
        rows_hold(&parts.problem, &b, Some(f.encoding().unsafe_row.0));
        for col in &parts.input_vars {
            assert_eq!((&a[col.0] + &b[col.0]) / q(2.), Q::zero());
        }
        assert_eq!((&a[heads[0].0] + &b[heads[0].0]) / q(2.), q(0.5));
        assert_eq!((&a[heads[head].0] + &b[heads[head].0]) / q(2.), q(1.));
    }
    // These actual graph mixtures prove membership for ANY valid affine row
    // on one target, the input and P, with arbitrary nonnegative coefficients.
    let a = pair(2, [1, 2]);
    let (dirs, bias) = residuals(&p, &a, end(), &mut || true).unwrap();
    assert_eq!(dirs, [vec![0.5, 1.], vec![0.5, -1.]]);
    let separate = dirs
        .iter()
        .zip(bias)
        .map(|(c, b)| {
            let v = dense::direction(c, b, &p).unwrap();
            dense::price(c, b, &v, &p, &[0., 0.]).unwrap()
        })
        .sum::<f64>();
    assert_eq!(separate, 1.5);
    assert!(1.5 <= separate);
    assert_eq!(
        price(&p, &dirs, &bias, &a.v, &[0., 0.], end(), &mut || true),
        Some(1.)
    );
    assert_eq!(oracle(&p, &a), q(1.));
    let checked = kernel::upper(&p, &a, end(), &mut || true).unwrap();
    assert!(q(checked) >= q(1.) && checked < 1.5);
}
#[test]
fn group_fresh_source_row_excludes_original_encoder_point_and_keeps_graph_atoms() {
    let f = toy();
    let p = f.source();
    let parts = f.parts();
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    let mut fake = graph(&f, &parts, &[q(0.), q(0.)], false);
    for (j, value) in [0.5, 1., 1.].into_iter().enumerate() {
        fake[heads[j].0] = q(value);
    }
    for col in &original.binary_vars {
        fake[col.0] = q(0.5);
    }
    fake[parts.output_vars[0].0] =
        q(f.b[1][0]) + (0..3).map(|j| q(f.w[1][j]) * &fake[heads[j].0]).sum::<Q>();
    rows_hold(&parts.problem, &fake, None);
    let ir = append_pure(&f, &parts, &p, &pair(2, [1, 2]));
    let row = ir.rows().last().unwrap();
    let lhs: Q = row.coeffs.iter().map(|&(j, a)| q(a) * &fake[j]).sum();
    assert_eq!(lhs, q(1.5));
    assert!(lhs > q(row.ub));
    for x in [-1., 0., 1.] {
        for y in [-1., 0., 1.] {
            for tie in [false, true] {
                rows_hold(
                    &ir,
                    &graph(&f, &parts, &[q(x), q(y)], tie),
                    Some(f.encoding().unsafe_row.0),
                );
            }
        }
    }
}
#[test]
fn group_joint_row_improves_original_relaxation_by_rational_half() {
    let f = toy();
    let p = f.source();
    let parts = f.parts();
    let strengthened = append_pure(&f, &parts, &p, &pair(2, [1, 2]));
    let before = lp(&parts.problem, &parts, f.encoding().unsafe_row.0);
    let after = lp(&strengthened, &parts, f.encoding().unsafe_row.0);
    assert!((before + 7. / 4.).abs() < 1e-8, "{before}");
    assert!((after + 5. / 4.).abs() < 1e-8, "{after}");
    assert!(after > before + 0.499999);
}
#[test]
fn group_quote_prepays_actual_adjacent_limits_and_maximum_single_or_pair_tail() {
    let p = pressure().source();
    assert_eq!(p.c[..62], vec![1.; 62]);
    assert_eq!(p.c[62..114], vec![-1.; 52]);
    let capture = primal_advice::quote(
        &p,
        primal_advice::RoleCounts {
            positive: 62,
            negative: 52,
        },
    )
    .unwrap()
    .debt()
    .0;
    assert_eq!(capture, 148_864);
    let a = quote(&p, 16, capture, 62, 52, 17).unwrap();
    assert_eq!(
        (a.total_work(), a.incremental_work(), a.positive_limit()),
        (652_832, 503_968, 17)
    );
    assert_eq!(a.checker_tail, a.base.alias_work.max(1152 * 31 + 128));
    assert_eq!(a.nominal_work, 64 * 17 * 31 + 576 * 31 + 64 * 17 + 1024);
    assert_eq!(a.bytes(), a.base.bytes + 128 * 31 + 4096);
    assert!(quote(&p, 16, capture, 62, 52, 18).is_none());
    let pair_tail = quote(&p, 16, 0, 62, 2, 1).unwrap();
    assert!(pair_tail.base.alias_work < 35_840);
    assert_eq!(pair_tail.checker_tail, 35_840);
    let singleton = quote(&p, 68, 153_664, 60, 54, 7).unwrap();
    assert_eq!(singleton.total_work(), 661_136);
    assert!(quote(&p, 68, 153_664, 60, 54, 8).is_none());
    assert!(quote(&p, 129, capture, 62, 52, 1).is_none());
    assert!(quote(&p, 16, capture, 62, 1, 1).is_none());
    assert!(quote(&p, 16, 664_696, 62, 52, 1).is_none());
    let other = p.clone();
    assert!(!a.matches(&other, 16));
    let mut wide = p.clone();
    wide.input_dim = 62;
    wide.w1 = vec![1.; 62 * 128];
    wide.lower = vec![-1.; 62];
    wide.upper = vec![1.; 62];
    assert!(quote(&wide, 16, 0, 62, 52, 1).is_none());
    assert!(a.matches(&p, 16));
}
#[test]
fn group_nominal_fallback_reuses_serialized_source_scaled_anchor_coefficients() {
    let mut f = pressure();
    f.w[1][62] = (-2f64).next_up().next_up();
    f.w[1][63] = (-2f64).next_up().next_up();
    let p = f.source();
    assert_eq!((p.c[62], p.c[63]), (-2., -2.));
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    let point = hint(&p, 0.5, 1.);
    let advice = dense::Advice {
        single: super::super::kernel::Proposal {
            head: 62,
            positive: vec![0],
            lambda: vec![1.],
            v: vec![0.; 30],
        },
        useful: false,
        partner: Some(63),
    };
    let a = proposal(
        row,
        &cuts,
        &point,
        t.group_quote().unwrap(),
        &advice,
        t.end,
        &mut || true,
    )
    .unwrap();
    assert_eq!(a.lambda, vec![2.]);
    assert_eq!(a.positive, vec![0]);
    assert_eq!(a.negative, [62, 63]);
    assert_eq!(a.v[0], 1.);
    assert!(a.v[1..].iter().all(|&v| v == 0.));
    let h = kernel::upper(&p, &a, t.end, &mut || true).unwrap();
    assert!((2. ..2. + 1e-10).contains(&h));
    let d = row.stats().component_rows.domination;
    assert_eq!(
        (
            d.advice_attempts,
            d.group_nominal_attempts,
            d.group_nominal_masks,
            d.support_attempts
        ),
        (1, 1, 4, 0)
    );
    assert!(resources::begin_advice(row, &t).is_none());
}
#[test]
fn group_useful_single_winner_is_retained_without_pair_check_or_new_capture() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    let mut point = hint(&p, 0., 1.);
    point.values[63] = 0.;
    let advice = dense::advice(
        row,
        &cuts,
        &point,
        t.dense_quote().unwrap(),
        t.end,
        &mut || true,
    )
    .unwrap();
    assert!(advice.useful);
    assert!(proposal(
        row,
        &cuts,
        &point,
        t.group_quote().unwrap(),
        &advice,
        t.end,
        &mut || true
    )
    .is_none());
    let token = super::super::kernel::check_with_aliases(
        row,
        &cuts,
        &t,
        advice.single,
        &point.negative,
        t.end,
        &mut || true,
    )
    .unwrap();
    assert_eq!(token.proposal.head, 62);
    let d = row.stats().component_rows.domination;
    assert_eq!(
        (
            d.support_attempts,
            d.completed_supports,
            d.group_checked_masks
        ),
        (1, 1, 0)
    );
    assert_eq!(cuts.proposal_lp_calls, 0);
    assert!(resources::support_attempt(row, &t, t.end).is_none());
}
#[test]
fn group_foreign_capture_ledger_and_extended_ticket_cannot_authorize_paid_advice() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let other = p.clone();
    let foreign = manual(&other);
    first(&foreign);
    let foreign_row = foreign.signed.as_ref().unwrap();
    let a = foreign.primal.as_ref().unwrap().lock().unwrap();
    assert!(a.point(&p, end(), &row.budget.state).is_none());
    assert!(resources::admit_group_point(row, &cuts, end(), Some(&a)).is_none());
    assert!(!row.stats().component_rows.domination.attempted);
    drop(a);
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let mut t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
    let old = t.end;
    t.end += Duration::from_nanos(1);
    assert!(resources::begin_advice(row, &t).is_none());
    t.end = old;
    assert!(resources::begin_advice(foreign_row, &t).is_none());
    assert!(resources::support_attempt(foreign_row, &t, t.end).is_none());
    resources::begin_advice(row, &t).unwrap();
    assert!(account
        .point(&p, t.end + Duration::from_secs(1), &row.budget.state)
        .is_none());
    assert_eq!(
        row.stats().component_rows.domination.capture_work_units,
        148_864
    );
}
#[test]
fn group_single_or_pair_support_capability_survives_counter_and_quota_resets() {
    for single_first in [false, true] {
        let p = pressure().source();
        let cuts = manual(&p);
        first(&cuts);
        let row = cuts.signed.as_ref().unwrap();
        let account = cuts.primal.as_ref().unwrap().lock().unwrap();
        let mut t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
        resources::begin_advice(row, &t).unwrap();
        row.budget
            .state
            .lock()
            .unwrap()
            .component_rows
            .domination
            .advice_attempts = 0;
        assert!(resources::begin_advice(row, &t).is_none());
        let a = pair(30, [62, 63]);
        let single = super::super::kernel::Proposal {
            head: 62,
            positive: vec![0],
            lambda: vec![1.],
            v: vec![0.; 30],
        };
        if single_first {
            assert!(
                super::super::kernel::check(row, &cuts, &t, single, t.end, &mut || true).is_some()
            );
        } else {
            assert!(kernel::check(row, &cuts, &t, a.clone(), t.end, &mut || true).is_some());
        }
        let work = row.stats().work_units;
        {
            let mut c = row.budget.state.lock().unwrap();
            c.component_rows.domination.support_attempts = 0;
            c.component_rows.domination.funded_targets = 2;
        }
        t.targets = 2;
        assert!(resources::support_attempt(row, &t, t.end).is_none());
        t.targets = 1;
        row.budget
            .state
            .lock()
            .unwrap()
            .component_rows
            .domination
            .funded_targets = 1;
        assert!(kernel::check(row, &cuts, &t, a, t.end, &mut || true).is_none());
        assert!(resources::support_attempt(row, &t, t.end).is_none());
        assert_eq!(row.stats().work_units, work);
        assert_eq!(row.stats().component_rows.domination.completed_supports, 1);
    }
}
#[test]
fn group_last_original_mask_refusal_keeps_paid_work_and_two_replay_limit() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let before = row.stats().work_units;
    let checks = row.budget.state.lock().unwrap().checks;
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
    assert_eq!(row.plan.components.len(), 16);
    assert_eq!(t.group_quote().unwrap().positive_limit(), 17);
    assert_eq!(
        row.stats()
            .component_rows
            .domination
            .total_optional_work_units,
        652_832
    );
    assert_eq!(row.stats().work_units, before + 503_968);
    assert_eq!(row.budget.state.lock().unwrap().checks, checks + 1);
    resources::begin_advice(row, &t).unwrap();
    let point = hint(&p, 0.5, 1.);
    let advice = dense::advice(
        row,
        &cuts,
        &point,
        t.dense_quote().unwrap(),
        t.end,
        &mut || true,
    )
    .unwrap();
    assert!(!advice.useful);
    assert_eq!((advice.single.head, advice.partner), (62, Some(63)));
    assert_ne!(
        row.plan
            .components
            .iter()
            .position(|c| c.negative.contains(&62)),
        row.plan
            .components
            .iter()
            .position(|c| c.negative.contains(&63))
    );
    let a = proposal(
        row,
        &cuts,
        &point,
        t.group_quote().unwrap(),
        &advice,
        t.end,
        &mut || true,
    )
    .unwrap();
    assert_eq!(a.lambda, vec![1.]);
    assert_eq!(a.v, pair(30, [62, 63]).v);
    let mut polls = 0;
    kernel::upper(&p, &a, t.end, &mut || {
        polls += 1;
        true
    })
    .unwrap();
    let work = row.stats().work_units;
    let mut seen = 0;
    assert!(kernel::check(row, &cuts, &t, a, t.end, &mut || {
        seen += 1;
        seen < polls
    })
    .is_none());
    assert_eq!(seen, polls);
    let d = row.stats().component_rows.domination;
    assert_eq!(
        (
            d.support_attempts,
            d.completed_supports,
            d.group_checked_masks,
            d.prepared_rows
        ),
        (1, 0, 0, 0)
    );
    assert_eq!(row.stats().work_units, work);
    assert_eq!(row.budget.state.lock().unwrap().checks, checks + 1);
    assert!(resources::support_attempt(row, &t, t.end).is_none());
    drop(account);
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_some());
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_none());
    assert_eq!((row.stats().source_replays, cuts.proposal_lp_calls), (2, 0));

    // The supplied strict point now traverses authenticated original capture,
    // binding, orchestration, fresh support, export and the actual commit.
    // No optimizing LP call or discovery claim is made for this manual hint.
    let f = pressure();
    let p = f.source();
    let cuts = manual_hint(&p);
    let mut parts = f.parts();
    let original_rows = parts.problem.num_rows();
    let row = cuts.signed.as_ref().unwrap();
    {
        let account = cuts.primal.as_ref().unwrap().lock().unwrap();
        let point = account
            .point(&p, row.budget.search_end, &row.budget.state)
            .unwrap();
        assert_eq!(
            (point.values[0], point.values[62], point.values[63]),
            (0.5, 1., 1.)
        );
    }
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let signed = report.signed.as_ref().unwrap();
    let d = &signed.component_rows.domination;
    assert!(d.group_allocation && d.admitted);
    assert_eq!(
        (
            d.advice_attempts,
            d.group_nominal_attempts,
            d.group_nominal_masks,
            d.support_attempts,
            d.completed_supports,
            d.group_checked_masks
        ),
        (1, 1, 4, 1, 1, 4)
    );
    assert_eq!(
        (
            d.prepared_rows,
            d.appended_rows,
            d.prepared_head_terms,
            d.prepared_terms
        ),
        (1, 1, 3, 4)
    );
    assert_eq!((d.group_anchor, d.group_partner), (Some(62), Some(63)));
    assert_eq!(
        (d.group_nominal_lhs, d.group_nominal_bound),
        (Some(1.5), Some(1.))
    );
    assert_eq!((report.proposal_lp_calls, signed.source_replays), (0, 1));
    assert!(parts.problem.num_rows() > original_rows);
    assert!(
        d.total_optional_work_units <= 664_696
            && signed.work_units <= MAX_WORK
            && signed.admitted_bytes <= MAX_BYTES
    );
    for value in [-1., 0., 1.] {
        rows_hold(
            &parts.problem,
            &graph(&f, &parts, &vec![q(value); 30], false),
            Some(f.encoding().unsafe_row.0),
        );
    }
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_some());
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_none());
    assert_eq!(row.stats().source_replays, 2);
}
#[test]
fn group_checked_token_seals_source_ledger_witness_signature_coefficients_and_clock() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let other = p.clone();
    let foreign = manual(&other);
    let foreign_row = foreign.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
    let mut token = kernel::check(row, &cuts, &t, pair(30, [62, 63]), t.end, &mut || true).unwrap();
    assert!(token.owned(row));
    let source = token.source;
    token.source ^= 1;
    assert!(!token.owned(row));
    token.source = source;
    let ledger = Arc::clone(&token.ledger);
    token.ledger = Arc::clone(&foreign_row.budget.state);
    assert!(!token.owned(row));
    token.ledger = ledger;
    let witness = Arc::clone(&token.witness);
    token.witness = Arc::new((*witness).clone());
    assert!(!token.owned(row));
    token.witness = witness;
    let signature = Arc::clone(&token.signature);
    token.signature = Arc::new((*signature).clone());
    assert!(!token.owned(row));
    token.signature = signature;
    let end = token.end;
    token.end += Duration::from_nanos(1);
    assert!(!token.owned(row));
    token.end = end;
    token.proposal.negative.swap(0, 1);
    assert!(!token.owned(row));
    token.proposal.negative.swap(0, 1);
    token.proposal.lambda[0] = 2.;
    assert!(!token.owned(row));
    token.proposal.lambda[0] = 1.;
    token.proposal.v[0] = f64::from_bits(token.proposal.v[0].to_bits() ^ 1);
    assert!(!token.owned(row));
    token.proposal.v[0] = f64::from_bits(token.proposal.v[0].to_bits() ^ 1);
    assert!(token.owned(row));
    assert!(!token.owned(foreign_row));
    drop(account);
    row.replay_inner(&cuts, row.authority(), &mut || true, false)
        .unwrap();
    assert!(!token.owned(row));
}
#[test]
fn group_final_export_cancellation_and_corrupt_mask_count_cannot_renew_export() {
    let mut final_poll = 0;
    for mode in 0..3 {
        let f = pressure();
        let p = f.source();
        let cuts = manual(&p);
        first(&cuts);
        let row = cuts.signed.as_ref().unwrap();
        let account = cuts.primal.as_ref().unwrap().lock().unwrap();
        let t = resources::admit_group_point(row, &cuts, end(), Some(&account)).unwrap();
        let token = kernel::check(row, &cuts, &t, pair(30, [62, 63]), t.end, &mut || true).unwrap();
        let parts = f.parts();
        let work = row.stats().work_units;
        if mode == 2 {
            row.budget
                .state
                .lock()
                .unwrap()
                .component_rows
                .domination
                .group_checked_masks = 3;
        }
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            let mut seen = 0;
            let result = kernel::export(row, &cuts, &parts, binding, &token, t.end, &mut || {
                seen += 1;
                mode != 1 || seen < final_poll
            });
            if mode == 0 {
                assert_eq!(result.unwrap().len(), 1);
                final_poll = seen;
                assert!(final_poll > 2);
            } else {
                assert!(result.is_none());
                if mode == 1 {
                    assert_eq!(seen, final_poll);
                    assert_eq!(row.stats().component_rows.domination.prepared_rows, 0);
                }
            }
            {
                let mut c = row.budget.state.lock().unwrap();
                c.component_rows.domination.export_attempts = 0;
                c.component_rows.domination.group_checked_masks = 4;
            }
            assert!(
                kernel::export(row, &cuts, &parts, binding, &token, t.end, &mut || true).is_none()
            );
        })
        .unwrap();
        assert_eq!(row.stats().work_units, work);
        assert_eq!(parts.problem.num_rows(), f.parts().problem.num_rows());
    }
}
#[test]
fn group_empty_mask_full_bias_nonfinite_overflow_and_malformed_support_refuse() {
    let mut p = toy().source();
    let a = pair(2, [1, 2]);
    p.lower = vec![1., -2.];
    p.upper = vec![2., -1.];
    p.b1 = vec![0.25, 0.75, -1.];
    let h = kernel::upper(&p, &a, end(), &mut || true).unwrap();
    assert!(q(h) >= oracle(&p, &a));
    p.w1.fill(0.);
    p.b1 = vec![0., -10., -10.];
    p.lower = vec![-1., -2.];
    p.upper = vec![1., 2.];
    let mut empty = a.clone();
    empty.v = vec![2., -3.];
    let h = kernel::upper(&p, &empty, end(), &mut || true).unwrap();
    assert!((8. ..8. + 1e-10).contains(&h));
    for mode in 0..6 {
        let mut bad = a.clone();
        match mode {
            0 => bad.negative = [1, 1],
            1 => {
                bad.positive = vec![0, 0];
                bad.lambda = vec![1., 1.];
            }
            2 => bad.lambda[0] = -1.,
            3 => bad.lambda[0] = 0.,
            4 => bad.v[0] = f64::NAN,
            _ => bad.positive[0] = 99,
        }
        assert!(kernel::upper(&p, &bad, end(), &mut || true).is_none());
    }
    p.c[0] = f64::NAN;
    assert!(kernel::upper(&p, &a, end(), &mut || true).is_none());
    p.c[0] = 1.;
    assert!(kernel::upper(
        &p,
        &a,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        &mut || { true }
    )
    .is_none());
    assert!(kernel::upper(&p, &a, end(), &mut || false).is_none());
    p.b1[1] = f64::MAX;
    p.c[1] = -2.;
    assert!(kernel::upper(&p, &a, end(), &mut || true).is_none());
}

#[test]
fn group_quote_rejects_inconsistent_totals_and_foreign_topology_before_debit() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let before = row.stats();
    let base = quote(&p, 16, 0, 62, 52, 17).unwrap();
    let capture = resources::MAX_DOMINATION_WORK
        .checked_sub(base.incremental_work())
        .unwrap();
    let exact = quote(&p, 16, capture, 62, 52, 17).unwrap();
    assert_eq!(
        exact.checked_total(&p, 16, capture),
        Some(resources::MAX_DOMINATION_WORK)
    );
    assert!(quote(&p, 16, capture.checked_add(1).unwrap(), 62, 52, 17).is_none());
    let other = p.clone();
    for field in 0..8 {
        let mut forged = quote(&p, 16, capture, 62, 52, 17).unwrap();
        match field {
            0 => forged.total -= 1,
            1 => forged.incremental += 1,
            2 => forged.base.total_work += 1,
            3 => forged.nominal_work += 1,
            4 => forged.checker_tail += 1,
            5 => forged.payload += 1,
            6 => forged.source = std::ptr::from_ref(&other) as usize,
            7 => forged.components += 1,
            _ => unreachable!(),
        }
        assert_eq!(forged.checked_total(&p, 16, capture), None);
        assert_eq!(row.stats().work_units, before.work_units);
        assert_eq!(row.stats().admitted_bytes, before.admitted_bytes);
    }
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    assert!(resources::admit_group_point(row, &cuts, end(), Some(&account)).is_some());
}
