// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fresh interval authority, complete masks, private seals and atomic export.
use super::*;
use nominal::test_fixture::{end, first, manual, pressure};
use num_rational::BigRational as Q;
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn pair() -> Proposal {
    let mut v = vec![0.; 30];
    v[0] = 0.5;
    Proposal {
        negative: [62, 63],
        positive: vec![0],
        lambda: vec![1.],
        v,
    }
}
fn oracle(p: &HullMarginProblem, a: &Proposal, tau: [f64; 4]) -> Q {
    (0..4)
        .flat_map(|mask| (0..1usize << p.input_dim).map(move |corner| (mask, corner)))
        .map(|(mask, corner)| {
            let x: Vec<_> = (0..p.input_dim)
                .map(|i| {
                    q(if corner & (1 << i) == 0 {
                        p.lower[i]
                    } else {
                        p.upper[i]
                    })
                })
                .collect();
            let affine = |j: usize| {
                q(p.b1[j]) + p.w_row(j).iter().zip(&x).map(|(&c, x)| q(c) * x).sum::<Q>()
            };
            let b = a
                .positive
                .iter()
                .zip(&a.lambda)
                .map(|(&j, &l)| q(l) * affine(j))
                .sum::<Q>();
            let active = (0..2)
                .filter(|t| mask & (1 << t) != 0)
                .map(|t| q(-p.c[a.negative[t]]) * affine(a.negative[t]))
                .sum::<Q>();
            active - q(tau[mask]) * b - a.v.iter().zip(&x).map(|(&c, x)| q(c) * x).sum::<Q>()
        })
        .max()
        .unwrap()
}
#[test]
fn branch_source_checker_rebuilds_all_four_bias_signed_box_supports_with_negative_upper() {
    // Supplied coefficients, not default discovery: the exact strict-family
    // fractional point is encoded in the ORIGINAL graph relaxation. Fresh
    // Source tau support excludes it while the best residual-split floor .75
    // (proved in steering controls) admits its .6 left-hand side.
    use super::super::super::super::test_fixture::{rows_hold, Fixture};
    use super::super::super::test_fixture::graph;
    let lifted = |x: f64| x.next_up().next_up();
    let f = Fixture {
        w: vec![
            vec![0., 0., 1., 1., 1., -1.],
            vec![lifted(1.), lifted(-1.), lifted(-1.), 0., 0., 0.],
        ],
        b: vec![vec![1., 0., 0.], vec![0.25, 0.]],
        dims: vec![2, 3, 2],
        input: vec![Bound::new(-1., 1.); 2],
        bounds: vec![vec![
            Bound::new(1., 1.),
            Bound::new(-2., 2.),
            Bound::new(-2., 2.),
        ]],
    };
    let source = f.source();
    assert_eq!(source.c, vec![1., -1., -1.]);
    let parts = f.parts();
    let a = Proposal {
        negative: [1, 2],
        positive: vec![0],
        lambda: vec![1.],
        v: vec![0.75, 0.],
    };
    let half = super::super::upper(&source, &a, end(), &mut || true).unwrap();
    let tight = upper(&source, &a, [1.; 4], end(), &mut || true).unwrap();
    assert!(half >= 0.75 && (0.25..0.2500000001).contains(&tight));
    let heads = binding::head_columns(&f.encoding(), &f.original()).unwrap();
    let mut fake = graph(&f, &parts, &[q(0.), q(0.)], false);
    fake[heads[1].0] = q(0.8);
    fake[heads[2].0] = q(0.8);
    for &j in &f.original().binary_vars {
        fake[j.0] = q(0.5);
    }
    fake[parts.output_vars[0].0] =
        q(0.25) + (0..3).map(|j| q(f.w[1][j]) * &fake[heads[j].0]).sum::<Q>();
    rows_hold(&parts.problem, &fake, None);
    let coeffs = vec![
        (heads[1], 1.),
        (heads[2], 1.),
        (heads[0], -1.),
        (parts.input_vars[0], -0.75),
    ];
    let lhs = coeffs.iter().map(|&(j, c)| q(c) * &fake[j.0]).sum::<Q>();
    assert!(lhs <= q(half) && lhs > q(tight));
    let mut strengthened = parts.problem.clone();
    strengthened.add_row(f64::NEG_INFINITY, tight, coeffs);
    for x in [-1., 0., 1.] {
        for y in [-1., 0., 1.] {
            for tie in [false, true] {
                rows_hold(
                    &strengthened,
                    &graph(&f, &parts, &[q(x), q(y)], tie),
                    Some(f.encoding().unsafe_row.0),
                );
            }
        }
    }

    let mut p = pressure().source();
    p.input_dim = 2;
    p.w1 = vec![0.; 2 * 128];
    p.lower = vec![-2., 1.];
    p.upper = vec![-1., 2.];
    p.w1[0] = 1.;
    p.w1[1] = -0.5;
    p.b1[0] = 0.75;
    p.w1[124] = 2.;
    p.w1[125] = 1.;
    p.b1[62] = -0.25;
    p.w1[126] = 1.;
    p.w1[127] = -1.;
    p.b1[63] = 0.5;
    let mut a = pair();
    a.v = vec![1., -0.25];
    let tau = [0.25, 0.75, 0.5, 1.];
    let h = upper(&p, &a, tau, end(), &mut || true).unwrap();
    let exact = oracle(&p, &a, tau);
    assert!(q(h) >= exact);
    assert!(q(h) - exact < q(1e-10));
    p.w1.fill(0.);
    p.b1.fill(0.);
    p.b1[0] = 1.;
    a.v = vec![0.; 2];
    let h = upper(&p, &a, [1.; 4], end(), &mut || true).unwrap();
    assert!((-1. ..0.).contains(&h));
    assert!(upper(&p, &a, [0.; 4], end(), &mut || true).unwrap() >= 0.);
    for bad in [f64::NAN, f64::INFINITY, -f64::MIN_POSITIVE, 1f64.next_up()] {
        let mut t = [0.5; 4];
        t[3] = bad;
        assert!(upper(&p, &a, t, end(), &mut || true).is_none());
    }
    p.w1[0] = f64::MAX;
    a.lambda[0] = 2.;
    assert!(upper(&p, &a, [1.; 4], end(), &mut || true).is_none());
}
#[test]
fn branch_source_token_seals_tau_kind_original_coefficients_and_all_owner_clocks() {
    let p = pressure().source();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    resources::select_branch_route(row, &t, false, true).unwrap();
    assert!(resources::begin_branch_tail(row, &t));
    let mut token = check(
        row,
        &cuts,
        &t,
        nominal::Proposal {
            affine: pair(),
            tau: [0., 0.5, 0.5, 1.],
        },
        t.end,
        &mut || true,
    )
    .unwrap();
    assert!(token.owned(row));
    for i in 0..4 {
        let saved = token.branch_tau.unwrap();
        token.branch_tau.as_mut().unwrap()[i] = 0.25;
        assert!(!token.owned(row));
        token.branch_tau = Some(saved);
    }
    let tau = token.branch_tau.take();
    assert!(!token.owned(row));
    token.branch_tau = tau;
    let e = token.end;
    token.end += Duration::from_nanos(1);
    assert!(!token.owned(row));
    token.end = e;
    let owner = Arc::clone(&token.witness);
    token.witness = Arc::new((*owner).clone());
    assert!(!token.owned(row));
    token.witness = owner;
    let owner = Arc::clone(&token.signature);
    token.signature = Arc::new((*owner).clone());
    assert!(!token.owned(row));
    token.signature = owner;
    let other = p.clone();
    let foreign = manual(&other);
    let foreign_row = foreign.signed.as_ref().unwrap();
    let owner = Arc::clone(&token.ledger);
    token.ledger = Arc::clone(&foreign_row.budget.state);
    assert!(!token.owned(row));
    token.ledger = owner;
    assert!(!token.owned(foreign_row));
    let source = token.source;
    token.source ^= 1;
    assert!(!token.owned(row));
    token.source = source;
    let saved = token.proposal.v[0];
    token.proposal.v[0] = saved.next_up();
    assert!(!token.owned(row));
    token.proposal.v[0] = saved;
    assert!(token.owned(row));
    drop(account);
    row.replay_inner(&cuts, row.authority(), &mut || true, false)
        .unwrap();
    assert!(!token.owned(row));
}
#[test]
fn branch_final_source_mask_cancellation_keeps_debit_checker_and_both_replay_limits() {
    let p = pressure().source();
    let a = pair();
    let mut total = 0;
    upper(&p, &a, [0., 0.5, 0.5, 1.], end(), &mut || {
        total += 1;
        true
    })
    .unwrap();
    let cuts = manual(&p);
    first(&cuts);
    let row = cuts.signed.as_ref().unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
    resources::begin_advice(row, &t).unwrap();
    resources::select_branch_route(row, &t, false, true).unwrap();
    resources::begin_branch_tail(row, &t);
    let work = row.stats().work_units;
    let mut seen = 0;
    assert!(check(
        row,
        &cuts,
        &t,
        nominal::Proposal {
            affine: a.clone(),
            tau: [0., 0.5, 0.5, 1.]
        },
        t.end,
        &mut || {
            seen += 1;
            seen < total
        }
    )
    .is_none());
    assert_eq!(seen, total);
    assert!(check(
        row,
        &cuts,
        &t,
        nominal::Proposal {
            affine: a,
            tau: [0., 0.5, 0.5, 1.]
        },
        t.end,
        &mut || true
    )
    .is_none());
    let d = row.stats().component_rows.domination;
    assert_eq!(
        (
            d.support_attempts,
            d.completed_supports,
            d.branch_tau_checked_masks
        ),
        (1, 0, 0)
    );
    assert_eq!(row.stats().work_units, work);
    assert!(row.budget.state.lock().unwrap().reserved > 0);
    drop(account);
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_some());
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_none());
}
#[test]
fn branch_final_export_poll_and_expired_caller_cannot_publish_or_restart_token() {
    let mut last = 0;
    for mode in 0..3 {
        let f = pressure();
        let p = f.source();
        let cuts = manual(&p);
        first(&cuts);
        let row = cuts.signed.as_ref().unwrap();
        let account = cuts.primal.as_ref().unwrap().lock().unwrap();
        let t = resources::admit_branch_point(row, &cuts, end(), Some(&account)).unwrap();
        resources::begin_advice(row, &t).unwrap();
        resources::select_branch_route(row, &t, false, true).unwrap();
        resources::begin_branch_tail(row, &t);
        let token = check(
            row,
            &cuts,
            &t,
            nominal::Proposal {
                affine: pair(),
                tau: [0., 0.5, 0.5, 1.],
            },
            t.end,
            &mut || true,
        )
        .unwrap();
        let parts = f.parts();
        let work = row.stats().work_units;
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            let e = if mode == 2 {
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("overflow when subtracting duration from instant")
            } else {
                t.end
            };
            let mut seen = 0;
            let result = export(row, &cuts, &parts, binding, &token, e, &mut || {
                seen += 1;
                mode != 1 || seen < last
            });
            if mode == 0 {
                assert_eq!(result.unwrap().len(), 1);
                last = seen;
            } else {
                assert!(result.is_none());
                assert_eq!(row.stats().component_rows.domination.prepared_rows, 0);
            }
            if mode == 1 {
                assert_eq!(seen, last);
            }
            if mode == 2 {
                assert_eq!(seen, 0);
            }
            row.budget
                .state
                .lock()
                .unwrap()
                .component_rows
                .domination
                .export_attempts = 0;
            assert!(export(row, &cuts, &parts, binding, &token, t.end, &mut || true).is_none());
        })
        .unwrap();
        assert_eq!(row.stats().work_units, work);
        assert_eq!(parts.problem.num_rows(), f.parts().problem.num_rows());
    }
}
