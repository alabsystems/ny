// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied legal allocations, with actual original-LP activity and Source checks.
//! This does not claim automatic discovery of the useful multi-selector row.
use super::*;

const E: f64 = 1. / 1024.;

pub(in crate::single_hidden_hull::mip_cuts::signed::portfolio) fn fixture(bias: f64) -> Fixture {
    Fixture {
        w: vec![
            vec![
                1.,
                0.,
                0.,
                0.,
                1.,
                0.,
                1.,
                0.,
                E,
                0.,
                1.,
                2. * E,
                1.,
                1.,
                0.,
            ],
            vec![2., 2., -1., -1., -1., 0., 0., 0., 0., 0.],
        ],
        b: vec![vec![-0.5, 0., 0., 0., -3.], vec![bias, 0.]],
        dims: vec![3, 5, 2],
        bounds: vec![vec![
            Bound::new(-1.5, 0.5),
            Bound::new(-1., 1.),
            Bound::new(-(1. + E) as f32, (1. + E) as f32),
            Bound::new(-(1. + 2. * E) as f32, (1. + 2. * E) as f32),
            Bound::new(-5., -1.),
        ]],
    }
}

fn plan(losing: bool) -> Plan {
    Plan {
        components: vec![
            Component {
                negative: vec![2, 3],
                positive: vec![0, 1],
                positive_units: vec![if losing { 1 } else { 2 }; 2],
                g: vec![0.; 3],
                alpha: if losing {
                    vec![0., 0., 0., 0.75, 0., 1., 1., 1.]
                } else {
                    vec![0.25, 0.25, 0.75, 0., 0.25, 0.75, 0.75, 0.5]
                },
            },
            Component {
                negative: vec![4],
                positive: if losing { vec![0, 1] } else { vec![] },
                positive_units: if losing { vec![1, 1] } else { vec![] },
                g: vec![0.; 3],
                alpha: if losing { vec![0.; 4] } else { vec![] },
            },
        ],
        leftover: vec![],
    }
}

pub(in crate::single_hidden_hull::mip_cuts::signed::portfolio) fn candidate(
    p: &HullMarginProblem,
    retain_cube: bool,
) -> HullRootMipCuts<'_> {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let mut cuts =
        propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).expect("one original root LP");
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert!(cuts.lp_primal_report().same_call && cuts.lp_primal_report().decoded);
    let mut b = Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO).unwrap();
    b.portfolio_enabled = true;
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let winner = plan(false);
    let cost = winner.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &winner, &b, b.search_end, &mut || true).unwrap();
    root_coupling::admit(&cuts, &winner, cost, &mut b).unwrap();
    component_rows::admit(&cuts, &winner, &b).unwrap();
    // Isolate losing Cube's original fractional-point/LP/AY comparison from
    // the newer winner Cell, without refunding any of its genuine admission.
    phase::cube::cell::omit_after_payment_for_test(&b);
    native_publication::fund(&cuts, &b).unwrap();
    b.admit_sharing().unwrap();
    let signature = Arc::new(signature(&cuts).collect());
    let loser = plan(true);
    let activity = phase::cube::portfolio_activity(&loser, &b).unwrap();
    let values = activity[0].expect("actual root LP has both negative-head activities");
    eprintln!("losing Cube actual relaxed activity: {:?}", &values[..2]);
    assert!((values[0] - 0.75).abs() < 1e-8);
    assert!((values[1] - 0.5).abs() < 1e-8);
    b.reserve_at_least(loser.cost(p).unwrap()).unwrap();
    let mut permit = admit(&cuts, &loser, &b, &signature, 1).unwrap();
    assert!(permit.conditional.is_some() && permit.cube.is_some());
    if !retain_cube {
        // Supplied control drops only the new capability after genuine payment.
        // There is no refund or alternate Source evaluation.
        permit.cube = None;
    }
    pay_final(&cuts, &b, loser.cost(p).unwrap(), Some(&mut permit)).unwrap();
    let losing = check_final(
        &cuts,
        &loser,
        &b,
        Some(permit),
        checked.root_lower,
        &mut || true,
    )
    .unwrap();
    assert!(losing.root_lower < checked.root_lower);
    assert!((checked.root_lower - (p.d - 1.25 - 3. * E)).abs() < 1e-9);
    assert!((losing.root_lower - (p.d - 1.75 - E)).abs() < 1e-9);
    cuts.signed = Some(row(&cuts, &winner, checked, cost, &b, &signature));
    cuts
}

fn fractional(f: &Fixture, parts: &MipParts) -> Vec<Q> {
    let mut values = f.point(parts, &[q(0.5), q(0.), q(1.)], 0);
    let heads = binding::head_columns(&f.encoding(), parts).unwrap();
    // Preserve the fixture's addition-then-multiply rounding order.
    let pair_sum = 1. + 2. * E;
    let half_sum = 0.5 * pair_sum;
    for (j, v) in [2f64.powi(-24), 0., 0.75 * (1. + E), half_sum, 0.]
        .into_iter()
        .enumerate()
    {
        values[heads[j].0] = q(v);
    }
    for (j, v) in [0.5, 0.5, 0.75, 0.5].into_iter().enumerate() {
        values[parts.binary_vars[j].0] = q(v);
    }
    for o in 0..2 {
        values[parts.output_vars[o].0] = q(f.b[1][o])
            + (0..5)
                .map(|j| q(f.w[1][5 * o + j]) * &values[heads[j].0])
                .sum::<Q>();
    }
    values
}

#[test]
fn losing_cube_same_paid_pass_separates_original_lp_and_ay() {
    let f = fixture(79. / 64.);
    let p = f.source();
    let old = candidate(&p, false);
    let cuts = candidate(&p, true);
    let mut baseline = f.original();
    old.append_to(&mut baseline, &f.encoding(), end()).unwrap();
    let mut parts = f.original();
    let appended = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let stats = cuts.mass_portfolio_report();
    eprintln!("losing Cube exported: {stats:?}");
    assert_eq!(stats.cube.completed_tables, 2);
    assert_eq!(stats.cube.collected_masks, 6);
    assert_eq!(stats.cube.appended_rows, 1);
    assert_eq!(stats.cube.appended_selector_terms, 2);
    assert_eq!(appended.proposal_lp_calls, 1);
    assert_eq!(appended.signed.as_ref().unwrap().source_replays, 1);
    assert_eq!(
        cuts.signed.as_ref().unwrap().stats().checked_masks,
        old.signed.as_ref().unwrap().stats().checked_masks
    );
    assert_eq!(
        cuts.signed
            .as_ref()
            .unwrap()
            .budget
            .state
            .lock()
            .unwrap()
            .checks,
        old.signed
            .as_ref()
            .unwrap()
            .budget
            .state
            .lock()
            .unwrap()
            .checks
    );
    for row in baseline.problem.rows() {
        assert!(parts.problem.rows().contains(row), "lost old row {row:?}");
    }
    let values = fractional(&f, &baseline);
    rows_hold(&baseline, &values, None, usize::MAX);
    let cube_row = parts.problem.rows().last().unwrap();
    let lhs: Q = cube_row
        .coeffs
        .iter()
        .map(|(j, a)| q(*a) * &values[*j])
        .sum();
    assert!(lhs < q(cube_row.lb));
    let rigorous = rational_certificate(&f, &parts, &stats);
    eprintln!("losing Cube exact stored-row certificate: {rigorous}");
    assert!(rigorous > q(1. / 16.));
    assert!(rigorous >= Q::new(619.into(), 9600.into()) - q(2f64.powi(-30)));
    let before = lp_floor(&baseline, f.encoding().unsafe_row.0, usize::MAX);
    let after = lp_floor(&parts, f.encoding().unsafe_row.0, usize::MAX);
    eprintln!("losing Cube actual original-LP lower: {before} -> {after}");
    assert!(before <= 0.);
    assert!(after > 1. / 16.);
    let result = crate::MipSolver::new(
        parts,
        crate::MipConfig {
            backend: crate::MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    )
    .check_feasibility()
    .unwrap();
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

fn rational_certificate(f: &Fixture, parts: &MipParts, stats: &SignedMassPortfolioReport) -> Q {
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    let selectors = [original.binary_vars[2].0, original.binary_vars[3].0];
    let rows = parts.problem.rows();
    let cube = rows.last().unwrap();
    let unary_end = rows.len() - stats.cube.appended_rows;
    let unaries: Vec<_> = rows[unary_end - stats.phase_appended_rows..unary_end]
        .iter()
        .filter(|r| {
            r.coeffs.iter().any(|(j, a)| *j == selectors[0] && *a != 0.)
                && !r.coeffs.iter().any(|(j, _)| *j == selectors[1])
        })
        .collect();
    assert_eq!(unaries.len(), 1);
    let mut coefficients = vec![Q::zero(); parts.num_cols];
    let mut rhs = Q::zero();
    weighted_row(
        unaries[0],
        Q::new(2.into(), 25.into()),
        &mut coefficients,
        &mut rhs,
    );
    weighted_row(
        cube,
        Q::new(14.into(), 75.into()),
        &mut coefficients,
        &mut rhs,
    );
    let original_row = |predicate: &dyn Fn(&RowSpec) -> bool| {
        let matching: Vec<_> = original
            .problem
            .rows()
            .iter()
            .filter(|r| predicate(r))
            .collect();
        assert_eq!(matching.len(), 1);
        matching[0]
    };
    let n0 = original_row(&|r| {
        r.ub == 1. + E
            && r.lb == f64::NEG_INFINITY
            && r.coeffs.contains(&(heads[2].0, 1.))
            && r.coeffs.contains(&(5, -1.))
            && r.coeffs.contains(&(selectors[0], 1. + E))
    });
    let n1 = original_row(&|r| {
        r.ub == 0.
            && r.lb == f64::NEG_INFINITY
            && r.coeffs.contains(&(heads[3].0, 1.))
            && r.coeffs.contains(&(selectors[1], -(1. + 2. * E)))
    });
    let p0 = original_row(&|r| {
        r.lb == 0.
            && r.ub == f64::INFINITY
            && r.coeffs.len() == 2
            && r.coeffs.contains(&(heads[0].0, 1.))
            && r.coeffs.contains(&(3, -1.))
    });
    let weight = Q::new(7.into(), 15.into());
    weighted_row(n0, -weight.clone(), &mut coefficients, &mut rhs);
    weighted_row(n1, -weight.clone(), &mut coefficients, &mut rhs);
    weighted_row(p0, weight, &mut coefficients, &mut rhs);
    for j in [0usize, 2] {
        let affine = &original.problem.rows()[j];
        assert_eq!(affine.lb, affine.ub);
        let column = 3 + j;
        let coefficient = affine.coeffs.iter().find(|(i, _)| *i == column).unwrap().1;
        let weight = -coefficients[column].clone() / q(coefficient);
        weighted_row(affine, weight, &mut coefficients, &mut rhs);
        assert!(coefficients[column].is_zero());
    }
    // Use raw network outputs, and price EVERY residual of the actual stored
    // rows over original column bounds. No ideal coefficient equality is assumed.
    let mut objective = vec![Q::zero(); parts.num_cols];
    for j in 0..5 {
        objective[heads[j].0] = q(f.w[1][j]) - q(f.w[1][5 + j]);
    }
    let mut lower = q(f.b[1][0]) - q(f.b[1][1]) + rhs;
    for (j, (want, got)) in objective.iter().zip(&coefficients).enumerate() {
        let residual = want - got;
        if residual.is_zero() {
            continue;
        }
        let column = &parts.problem.cols()[j];
        assert!(
            column.lb.is_finite() && column.ub.is_finite(),
            "unpriced column{j}"
        );
        lower += (&residual * q(column.lb)).min(&residual * q(column.ub));
    }
    lower
}

#[test]
fn losing_cube_preserves_unsafe_and_closed_zero_original_graphs() {
    for bias in [0.5 + 2. * E, 0.5 + 3. * E] {
        let f = fixture(bias);
        let p = f.source();
        let cuts = candidate(&p, true);
        let mut parts = f.original();
        cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        assert_eq!(cuts.mass_portfolio_report().cube.appended_rows, 1);
        for zero_mask in 0..4 {
            let values = f.point(&parts, &[q(0.5), q(0.), q(1.)], zero_mask);
            let margin = &values[parts.output_vars[0].0] - &values[parts.output_vars[1].0];
            assert_eq!(margin, q(bias - 0.5 - 3. * E));
            assert!(margin <= Q::zero());
            rows_hold(&parts, &values, None, usize::MAX);
        }
    }
}
