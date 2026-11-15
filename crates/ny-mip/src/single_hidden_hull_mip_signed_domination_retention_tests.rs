// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Paid-original-support, binding, clock and publication controls; unexecuted.
use super::test_fixture::*;
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
fn proposal(p: &HullMarginProblem) -> kernel::Proposal {
    kernel::Proposal {
        head: (0..p.hidden_dim).find(|&j| p.c[j] < 0.).unwrap(),
        positive: vec![(0..p.hidden_dim).find(|&j| p.c[j] > 0.).unwrap()],
        lambda: vec![1.],
        v: vec![0.; p.input_dim],
    }
}
#[test]
fn signed_domination_signed_translated_and_degenerate_boxes_price_both_branches() {
    let f = Fixture::sparse(-0.25);
    let mut p = f.source();
    p.w1[4] = 1.;
    p.w1[5] = 0.; // a=b=x, so the exact residual is zero.
    let mut a = proposal(&p);
    a.v = vec![1., 0.];
    for (l, u, want) in [(-2., -1., 2.), (1., 2., -1.), (1., 1., -1.)] {
        p.lower = vec![l, 0.];
        p.upper = vec![u, 0.];
        let h = kernel::upper(&p, &a, end(), &mut || true).unwrap();
        assert!(h >= want && (h - want).abs() < 1e-12, "H={h} wanted{want}");
    }
    p.lower = vec![-2., -0.5];
    p.upper = vec![3., 2.];
    p.w1[4] = 0.75;
    p.w1[5] = -0.5;
    p.b1[2] = 0.125;
    a.v = vec![0.25, -0.125];
    let h = kernel::upper(&p, &a, end(), &mut || true).unwrap();
    for x in [-2., 0., 3.] {
        for y in [-0.5, 0., 2.] {
            let left = (q(0.75) * q(x) - q(0.5) * q(y) + q(0.125)).max(Q::zero())
                - q(x).max(Q::zero())
                - q(0.25) * q(x)
                + q(0.125) * q(y);
            assert!(left <= q(h));
        }
    }
}
#[test]
fn signed_domination_original_residual_cancellation_subnormal_and_overflow_are_priced() {
    let f = Fixture::sparse(-0.25);
    let mut p = f.source();
    let mut a = proposal(&p);
    p.lower = vec![1., 0.];
    p.upper = p.lower.clone();
    let eps = f64::EPSILON;
    p.w1[0] = 1. - eps;
    p.w1[1] = 0.;
    p.w1[4] = 1.;
    p.w1[5] = 0.;
    a.lambda[0] = 1. + eps;
    let exact = q(1.) - q(1. + eps) * q(1. - eps);
    assert!(exact > Q::zero());
    assert_eq!(1. - (1. + eps) * (1. - eps), 0.);
    let h = kernel::upper(&p, &a, end(), &mut || true).unwrap();
    assert!(q(h) >= exact, "rounded zero is not the original residual");
    let tiny = f64::from_bits(1);
    p.w1[0] = -tiny;
    p.w1[4] = 0.;
    a.lambda[0] = 0.5;
    let h = kernel::upper(&p, &a, end(), &mut || true).unwrap();
    assert!(
        q(h) >= q(tiny) / q(2.),
        "subnormal support cannot be silently zero"
    );
    p.w1[0] = f64::MAX;
    a.lambda[0] = 2.;
    assert!(kernel::upper(&p, &a, end(), &mut || true).is_none());
}
#[test]
fn signed_domination_malformed_supports_and_nonfinite_box_decline_without_authority() {
    let f = Fixture::sparse(-0.25);
    let mut p = f.source();
    let a = proposal(&p);
    for lambda in [-1., f64::NAN, f64::INFINITY, 0.] {
        let mut bad = a.clone();
        bad.lambda[0] = lambda;
        assert!(kernel::upper(&p, &bad, end(), &mut || true).is_none());
    }
    for change in 0..5 {
        let mut bad = a.clone();
        match change {
            0 => bad.positive = vec![0, 0],
            1 => bad.positive[0] = bad.head,
            2 => bad.positive[0] = usize::MAX,
            3 => bad.v.pop().map(|_| ()).unwrap(),
            _ => bad.head = usize::MAX,
        }
        if change == 0 {
            bad.lambda = vec![1., 1.];
        }
        assert!(kernel::upper(&p, &bad, end(), &mut || true).is_none());
    }
    p.lower[0] = f64::NEG_INFINITY;
    assert!(kernel::upper(&p, &a, end(), &mut || true).is_none());
    assert!(kernel::upper(&f.source(), &a, end(), &mut || false).is_none());
    assert!(kernel::upper(
        &f.source(),
        &a,
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("overflow when subtracting duration from instant"),
        &mut || true
    )
    .is_none());
}
#[test]
fn signed_domination_whole_quote_keeps_exact_both_publication_holds_and_old_scalar() {
    let f = Fixture::large(0.25);
    let p = f.source();
    let c = cuts(&p, &[1, 1, 0]);
    let row = c.signed.as_ref().unwrap();
    let first = row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let before = {
        let s = row.budget.state.lock().unwrap();
        (
            s.work,
            s.reserved,
            s.checks,
            s.component_rows.work_units,
            s.component_rows.admitted_bytes,
        )
    };
    let expected = resources::quote(row, &c, 24, 0).unwrap();
    let t = resources::admit(row, &c, end()).unwrap();
    assert_eq!((t.targets, t.sparse), (24, 0));
    {
        let s = row.budget.state.lock().unwrap();
        assert_eq!(s.work, before.0 + expected.0);
        assert_eq!(s.reserved, before.1);
        assert_eq!(s.component_reserved, 0);
        assert_eq!(s.checks, before.2 + 1);
        assert_eq!(s.component_rows.work_units, before.3 + expected.0);
        assert_eq!(s.component_rows.admitted_bytes, before.4 + expected.1);
        assert_eq!(s.component_rows.domination.work_units, expected.0);
        assert!(s.work + s.reserved <= MAX_WORK);
    }
    assert!(expected.0 <= resources::MAX_DOMINATION_WORK);
    let token = kernel::check(row, &c, &t, proposal(&p), end(), &mut || true).unwrap();
    assert!(token.owned(row));
    assert!(resources::admit(row, &c, end()).is_none());
    let paid = row.budget.state.lock().unwrap().work;
    let second = row.replay_inner(&c, end(), &mut || true, false).unwrap();
    assert_eq!(first.root_lower.to_bits(), second.root_lower.to_bits());
    assert_eq!(first.constant.to_bits(), second.constant.to_bits());
    assert_eq!(first.g, second.g);
    {
        let s = row.budget.state.lock().unwrap();
        assert_eq!(s.work, paid + row.cost);
        assert_eq!(s.reserved, 0);
        assert!(s.work <= MAX_WORK);
    }
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.stats().source_replays, 2);
}
#[test]
fn signed_domination_unfunded_payload_checks_and_expiry_refuse_once_before_new_work() {
    for mode in 0..4 {
        let f = Fixture::sparse(-0.25);
        let p = f.source();
        let c = cuts(&p, &[2]);
        let row = c.signed.as_ref().unwrap();
        row.replay_inner(&c, end(), &mut || true, false).unwrap();
        {
            let mut s = row.budget.state.lock().unwrap();
            match mode {
                0 => s.work = MAX_WORK - s.reserved - s.component_reserved,
                1 => s.shared_bytes = MAX_BYTES,
                2 => s.checks = MAX_CHECKS,
                _ => (),
            }
        }
        let before = {
            let s = row.budget.state.lock().unwrap();
            (s.work, s.reserved, s.checks)
        };
        let cutoff = if mode == 3 {
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("overflow when subtracting duration from instant")
        } else {
            end()
        };
        assert!(resources::admit(row, &c, cutoff).is_none());
        assert!(resources::admit(row, &c, end()).is_none());
        let s = row.budget.state.lock().unwrap();
        assert_eq!((s.work, s.reserved, s.checks), before);
        let d = &s.component_rows.domination;
        assert!(d.attempted);
        assert!(!d.admitted);
        assert_eq!(
            (
                d.work_units,
                d.advice_attempts,
                d.support_attempts,
                d.completed_supports,
                d.prepared_rows
            ),
            (0, 0, 0, 0, 0)
        );
    }
}
#[test]
fn signed_domination_mutated_ticket_quota_clock_and_repeated_advice_cannot_renew_work() {
    for mode in 0..3 {
        let f = Fixture::sparse(-0.25);
        let p = f.source();
        let c = cuts(&p, &[2]);
        let row = c.signed.as_ref().unwrap();
        row.replay_inner(&c, end(), &mut || true, false).unwrap();
        let mut t = resources::admit(row, &c, end()).unwrap();
        match mode {
            0 => t.targets = usize::MAX,
            1 => t.end += Duration::from_secs(1),
            _ => t.sparse += 1,
        }
        assert!(resources::support_attempt(row, &t, end()).is_none());
        assert!(resources::begin_advice(row, &t).is_none());
        assert_eq!(
            row.budget
                .state
                .lock()
                .unwrap()
                .component_rows
                .domination
                .support_attempts,
            0
        );
    }
    let f = Fixture::sparse(-0.25);
    let p = f.source();
    let c = cuts(&p, &[2]);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let t = resources::admit(row, &c, end()).unwrap();
    resources::begin_advice(row, &t).unwrap();
    assert!(resources::begin_advice(row, &t).is_none());
    assert_eq!(t.targets, 1);
    assert!(resources::support_attempt(row, &t, end()).is_some());
    assert!(resources::support_attempt(row, &t, end()).is_none());
    let s = row.budget.state.lock().unwrap();
    assert_eq!(s.component_rows.domination.advice_attempts, 1);
    assert_eq!(s.component_rows.domination.support_attempts, 1);
    assert_eq!(s.component_rows.domination.completed_supports, 0);
}
#[test]
fn signed_domination_checked_owner_clock_support_bits_and_foreign_binding_refuse() {
    for mode in 0..6 {
        let f = Fixture::sparse(-0.25);
        let p = f.source();
        let c = cuts(&p, &[2]);
        let row = c.signed.as_ref().unwrap();
        row.replay_inner(&c, end(), &mut || true, false).unwrap();
        let t = resources::admit(row, &c, end()).unwrap();
        let mut token = kernel::check(row, &c, &t, proposal(&p), end(), &mut || true).unwrap();
        match mode {
            0 => token.end += Duration::from_secs(1),
            1 => token.proposal.lambda[0] = 2.,
            2 => token.upper = f64::INFINITY,
            3 => token.signature = Arc::new((*token.signature).clone()),
            4 => token.witness = Arc::new((*token.witness).clone()),
            _ => token.ledger = Arc::new(std::sync::Mutex::new(Counts::default())),
        }
        assert!(!token.owned(row));
        let parts = f.parts();
        let before = parts.problem.num_rows();
        let result = binding::with_phase_binding_test(&c, &parts, &f.encoding(), |binding| {
            kernel::export(row, &c, &parts, binding, &[token], end(), &mut || true)
        })
        .unwrap();
        assert!(result.is_none());
        assert_eq!(parts.problem.num_rows(), before);
        assert_eq!(
            row.budget
                .state
                .lock()
                .unwrap()
                .component_rows
                .domination
                .prepared_rows,
            0
        );
    }
    let f = Fixture::sparse(-0.25);
    let p = f.source();
    let c = cuts(&p, &[2]);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let t = resources::admit(row, &c, end()).unwrap();
    let foreign_source = f.source();
    let foreign = cuts(&foreign_source, &[2]);
    assert!(kernel::check(row, &foreign, &t, proposal(&p), end(), &mut || true).is_none());
    let mut altered = Fixture::sparse(-0.25);
    altered.w[0][0] += EPS;
    let parts = altered.parts();
    let mut called = false;
    let result =
        binding::with_phase_binding_test(&c, &parts, &altered.encoding(), |_| called = true);
    assert!(result.is_none() && !called);
}
#[test]
fn signed_domination_paid_partial_export_is_atomic_and_unfunded_joint_setup_is_skipped() {
    let f = Fixture::large(0.25);
    let p = f.source();
    let c = cuts(&p, &[1, 1, 0]);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let t = resources::admit(row, &c, end()).unwrap();
    let a = proposal(&p);
    let mut b = a.clone();
    b.head = 2;
    let checked = vec![
        kernel::check(row, &c, &t, a, end(), &mut || true).unwrap(),
        kernel::check(row, &c, &t, b, end(), &mut || true).unwrap(),
    ];
    let parts = f.parts();
    let before = parts.problem.num_rows();
    let mut polls = 0;
    let first_row_polls = 1 + 1 + p.input_dim;
    let result = binding::with_phase_binding_test(&c, &parts, &f.encoding(), |binding| {
        kernel::export(row, &c, &parts, binding, &checked, end(), &mut || {
            polls += 1;
            polls <= first_row_polls
        })
    })
    .unwrap();
    assert!(result.is_none());
    assert_eq!(polls, first_row_polls + 1);
    assert_eq!(parts.problem.num_rows(), before);
    {
        let s = row.budget.state.lock().unwrap();
        let d = &s.component_rows.domination;
        assert_eq!(d.completed_supports, 2);
        assert_eq!(d.prepared_rows, 0);
        assert!(d.work_units > 0);
    }
    assert!(resources::admit(row, &c, end()).is_none());
    {
        let mut s = row.budget.state.lock().unwrap();
        s.work = MAX_WORK - s.reserved - s.component_reserved;
    }
    let paid = row.budget.state.lock().unwrap().work;
    assert!(!allow_joint(row, &c));
    let s = row.budget.state.lock().unwrap();
    assert_eq!(s.work, paid);
    assert!(s.component_rows.domination.joint_skipped_unfunded);
    assert!(!s.component_rows.joint.attempted);
}

#[test]
fn signed_domination_stable_active_original_free_head_column_is_authenticated_and_exported() {
    let mut f = Fixture::sparse(-0.25);
    f.b[0][0] = 2.;
    f.bounds[0][0] = Bound::new(1., 3.);
    let p = f.source();
    let c = cuts(&p, &[2]);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let t = resources::admit(row, &c, end()).unwrap();
    let token = kernel::check(row, &c, &t, proposal(&p), end(), &mut || true).unwrap();
    let parts = f.parts();
    let before = parts.problem.num_rows();
    let rows = binding::with_phase_binding_test(&c, &parts, &f.encoding(), |binding| {
        let head = binding.head_column(0).unwrap();
        assert_eq!(parts.problem.cols()[head.0].lb, f64::NEG_INFINITY);
        assert!(
            binding.column(0).is_none(),
            "head mapping is not negative-selector mapping"
        );
        let rows = kernel::export(row, &c, &parts, binding, &[token], end(), &mut || true).unwrap();
        assert!(rows[0].coeffs.contains(&(head.0, -1.)));
        rows
    })
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(parts.problem.num_rows(), before);
    for x in [-1., 0., 1.] {
        for y in [-1., 0., 1.] {
            let v = point(&f, &parts, &[q(x), q(y)], None, 0);
            for r in &rows {
                let lhs: Q = r.coeffs.iter().map(|(j, a)| q(*a) * &v[*j]).sum();
                assert!(lhs <= q(r.ub));
            }
        }
    }
    assert_eq!(row.stats().component_rows.domination.prepared_rows, 1);
}
#[test]
fn signed_domination_final_original_support_poll_refuses_after_all_coordinates_without_refund() {
    let f = Fixture::large(0.25);
    let p = f.source();
    let c = cuts(&p, &[1, 1, 0]);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let t = resources::admit(row, &c, end()).unwrap();
    let before = {
        let s = row.budget.state.lock().unwrap();
        (s.work, s.reserved, s.checks)
    };
    let a = proposal(&p);
    let final_poll = 1 + a.positive.len() + p.input_dim + 1;
    let mut polls = 0;
    assert!(kernel::check(row, &c, &t, a, end(), &mut || {
        polls += 1;
        polls < final_poll
    })
    .is_none());
    assert_eq!(polls, final_poll);
    let s = row.budget.state.lock().unwrap();
    assert_eq!((s.work, s.reserved, s.checks), before);
    let d = &s.component_rows.domination;
    assert_eq!(
        (
            d.support_attempts,
            d.completed_supports,
            d.prepared_rows,
            d.export_attempts
        ),
        (1, 0, 0, 0)
    );
    assert!(d.work_units > 0);
    assert!(s.work + s.reserved <= MAX_WORK);
}
#[test]
fn signed_domination_and_joint_share_one_absolute_window_and_cannot_restart_after_expiry() {
    let f = Fixture::large(0.25);
    let p = f.source();
    let c = cuts(&p, &[1, 1, 0]);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let parts = f.parts();
    let mut expired = false;
    let before = parts.problem.num_rows();
    let rows = binding::with_phase_binding_test(&c, &parts, &f.encoding(), |binding| {
        joint::prepare(row, &c, &parts, binding, end(), &mut || {
            if !expired
                && row
                    .budget
                    .state
                    .lock()
                    .unwrap()
                    .component_rows
                    .domination
                    .admitted
            {
                // Deliberately consume the ONE existing 50ms optional window.
                // A fresh joint-family window would incorrectly permit setup below.
                std::thread::sleep(Duration::from_millis(60));
                expired = true;
            }
            true
        })
    })
    .unwrap();
    assert!(expired && rows.is_empty());
    assert_eq!(parts.problem.num_rows(), before);
    let paid = {
        let s = row.budget.state.lock().unwrap();
        assert!(s.component_rows.domination.admitted);
        assert!(s.component_rows.domination.work_units > 0);
        assert_eq!(s.component_rows.joint.producer_attempts, 1);
        assert_eq!(s.component_rows.joint.work_units, 0);
        s.work
    };
    let again = binding::with_phase_binding_test(&c, &parts, &f.encoding(), |binding| {
        joint::prepare(row, &c, &parts, binding, end(), &mut || true)
    })
    .unwrap();
    assert_eq!(again, [] as [RowSpec; 0]);
    assert_eq!(row.budget.state.lock().unwrap().work, paid);
}
#[test]
fn signed_domination_generic_30d_128h_adaptive_nine_quote_preserves_publication_and_residual_credit(
) {
    // Synthetic valid Source/partition with the engineering shape of the target;
    // its injected free credit is a conditional ledger control, not a benchmark.
    let (d, h) = (30, 128);
    let mut coeff = vec![0.; h];
    coeff[..60].fill(1.);
    coeff[60..114].fill(-1.);
    let mut w = vec![0.; d * h];
    for j in 0..h {
        w[j * d] = 1.;
    }
    let mut p = HullMarginProblem::new(
        w,
        vec![0.; h],
        d,
        &coeff,
        &[-1.],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    p.c = coeff;
    p.d = -1.;
    let negative_counts = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6];
    let positive_counts = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3];
    let (mut n, mut a) = (60, 0);
    let mut components = Vec::new();
    for (&k, &count) in negative_counts.iter().zip(&positive_counts) {
        let positive: Vec<_> = (a..a + count).collect();
        a += count;
        components.push(Component {
            negative: (n..n + k).collect(),
            positive_units: vec![2; count],
            alpha: vec![0.5; (1usize << k) * count],
            positive,
            g: vec![0.; d],
        });
        n += k;
    }
    assert_eq!((n, a), (114, 60));
    let c = cuts_with_plan(
        &p,
        Plan {
            components,
            leftover: vec![],
        },
    );
    let row = c.signed.as_ref().unwrap();
    assert_eq!(row.budget.dimensions, (30, 128, 60));
    assert_eq!(c.planes.len(), 54);
    assert_eq!(row.plan.components.len(), 16);
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    {
        let mut s = row.budget.state.lock().unwrap();
        s.work = MAX_WORK - s.reserved - s.component_reserved - 1_154_224;
    }
    let before = {
        let s = row.budget.state.lock().unwrap();
        (s.work, s.reserved, s.checks)
    };
    let width = d + 1;
    let setup = 64 * h * width + 16 * 54 * 60 + 128 * (h + 1) + 8192 + 32 * 16 * 16 + 128 * 16;
    let expected = setup + 64 * width * (4 * 9 + 16 * 2) + 128 * width * 5 * 9;
    assert_eq!(expected, 654_208);
    assert_eq!(resources::quote(row, &c, 9, 2).unwrap().0, expected);
    assert!(resources::quote(row, &c, 10, 2).unwrap().0 > resources::MAX_DOMINATION_WORK);
    let t = resources::admit(row, &c, end()).unwrap();
    assert_eq!((t.targets, t.sparse), (9, 2));
    {
        let s = row.budget.state.lock().unwrap();
        assert_eq!(s.work, before.0 + expected);
        assert_eq!(s.reserved, before.1);
        assert_eq!(s.checks, before.2 + 1);
        assert_eq!(resources::free(&s), Some(500_016));
        assert!(
            row.budget.admitted_bytes
                + s.shared_bytes
                + s.direction.cache_bytes
                + s.child.cache_bytes
                + s.component_rows.admitted_bytes
                <= MAX_BYTES
        );
    }
    // The separately observed old setup + cheapest pair quote cannot fit this
    // remainder. These constants are historical inputs, not this fixture's timing.
    assert_eq!((490_872usize + 86_912).checked_sub(500_016), Some(77_768));
    assert!(490_872 + 86_912 > resources::free(&row.budget.state.lock().unwrap()).unwrap());
    let paid = row.budget.state.lock().unwrap().work;
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let s = row.budget.state.lock().unwrap();
    assert_eq!(s.work, paid + row.cost);
    assert_eq!((s.reserved, s.source_replays), (0, 2));
    assert_eq!(resources::free(&s), Some(500_016));
    assert_eq!(s.child.attempted_calls, 0);
    assert!(resources::free(&s).unwrap() > 0);
    assert!(
        s.work <= MAX_WORK,
        "residual credit remains an opportunity, not a claimed child bound"
    );
}
