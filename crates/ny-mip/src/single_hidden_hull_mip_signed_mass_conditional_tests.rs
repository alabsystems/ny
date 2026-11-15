// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Automatic discovery and exact original-row separation, without supplied plans.
use super::*;

pub(in crate::single_hidden_hull::mip_cuts::signed::portfolio) fn fixture() -> Fixture {
    let (d, h) = (30, 18);
    let mut first = vec![0.; d * h];
    let mut biases = vec![0.; h];
    let mut bounds = vec![Bound::new(-1., 1.); h];
    for j in 0..h {
        first[j * d + d - 1] = 1.;
    }
    first[2 * d - 1] = -1.;
    biases[1] = -1. / 64.;
    bounds[1] = Bound::new(-65. / 64., 63. / 64.);
    for i in 0..5 {
        let j = 3 + i;
        let m = 2. + i as f64 / 64.;
        first[j * d + d - 1] = m;
        biases[j] = -m - 0.25;
        bounds[j] = Bound::new((-2. * m - 0.25) as f32, -0.25);
    }
    let mut output = vec![0.; 2 * h];
    output[0] = 28.;
    output[1] = 1.;
    output[2] = -97. / 8.;
    output[3..h].fill(-1.);
    Fixture {
        w: vec![first, output],
        b: vec![biases, vec![5. - 1. / 512., 0.]],
        dims: vec![d, h, 2],
        bounds: vec![bounds],
    }
}

pub(in crate::single_hidden_hull::mip_cuts::signed::portfolio) fn candidate(
    p: &HullMarginProblem,
) -> HullRootMipCuts<'_> {
    let attempt =
        propose_root_mip_cuts_with_conditional_mass_portfolio_attempt(p, end(), Duration::ZERO);
    assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
    let cuts = attempt.cuts.unwrap();
    let r = cuts.mass_portfolio_report();
    assert!(r.retained && r.phase_admitted, "{r:?}");
    assert_eq!(r.phase_collections, 1);
    assert!(r.candidate_lower.unwrap() < r.scalar_threshold.unwrap());
    assert!((r.candidate_lower.unwrap() + 17. / 512.).abs() < 1e-9);
    assert!((r.scalar_threshold.unwrap() + 1. / 512.).abs() < 1e-9);
    // Preserve the earlier losing-conditional separation and exact-row controls
    // independently of the newer winner Cell. Production routing is tested below.
    phase::cube::cell::omit_after_payment_for_test(&cuts.signed.as_ref().unwrap().budget);
    cuts
}

fn fake(f: &Fixture, parts: &MipParts) -> Vec<Q> {
    let heads = binding::head_columns(&f.encoding(), parts).unwrap();
    let mut values = vec![Q::zero(); parts.num_cols];
    for j in 0..18 {
        values[30 + j] = q(f.b[0][j]);
    }
    values[heads[0].0] = q(1.) / q(28.);
    values[heads[2].0] = q(8.) / q(97.);
    for j in 8..18 {
        values[heads[j].0] = q(0.5);
    }
    for var in &parts.binary_vars {
        values[var.0] = q(0.5);
    }
    values[parts.output_vars[0].0] = q(-1. / 512.);
    values
}

fn unique_row<'a>(rows: &'a [RowSpec], predicate: &dyn Fn(&RowSpec) -> bool) -> &'a RowSpec {
    let matches: Vec<_> = rows.iter().filter(|r| predicate(r)).collect();
    assert_eq!(matches.len(), 1, "ambiguous certificate row: {matches:?}");
    matches[0]
}

fn rational_certificate(
    f: &Fixture,
    p: &HullMarginProblem,
    parts: &MipParts,
    original: &MipParts,
    stats: &SignedMassPortfolioReport,
) -> Q {
    let heads = binding::head_columns(&f.encoding(), original).unwrap();
    let selector = original.binary_vars[2].0;
    let rows = parts.problem.rows();
    let start = rows.len() - stats.appended_rows;
    let phase_start = rows.len() - stats.phase_appended_rows;
    let winning = &rows[original.problem.num_rows()..start];
    let losing = &rows[start..phase_start];
    let phases = &rows[phase_start..];
    let lift = unique_row(phases, &|r| {
        r.coeffs.contains(&(heads[2].0, 2. * p.c[2]))
            && r.coeffs.iter().any(|(j, a)| *j == selector && *a < 0.)
    });
    let winner = unique_row(winning, &|r| {
        r.coeffs.contains(&(heads[0].0, 2. * p.c[0]))
            && r.coeffs.contains(&(heads[2].0, 2. * p.c[2]))
            && !r.coeffs.iter().any(|(j, _)| *j == selector)
    });
    let loser_b = unique_row(losing, &|r| {
        r.coeffs.contains(&(heads[0].0, p.c[0]))
            && !r.coeffs.iter().any(|(j, _)| *j == heads[2].0)
            && !r
                .coeffs
                .iter()
                .any(|(j, _)| original.binary_vars.iter().any(|v| v.0 == *j))
    });
    let b_head = (8..18)
        .find(|j| loser_b.coeffs.contains(&(heads[*j].0, 2. * p.c[*j])))
        .unwrap();
    let mut combined = vec![Q::zero(); parts.num_cols];
    let mut rhs = Q::zero();
    weighted_row(lift, Q::new(97.into(), 211.into()), &mut combined, &mut rhs);
    weighted_row(
        winner,
        Q::new(17.into(), 844.into()),
        &mut combined,
        &mut rhs,
    );
    weighted_row(loser_b, q(0.5), &mut combined, &mut rhs);
    for j in (8..18).filter(|j| *j != b_head) {
        let row = unique_row(winning, &|r| {
            r.coeffs.len() == 2
                && r.coeffs.contains(&(heads[j].0, 2. * p.c[j]))
                && r.coeffs.iter().any(|(i, _)| *i == parts.input_vars[29].0)
        });
        weighted_row(row, q(0.5), &mut combined, &mut rhs);
    }
    let upper = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.ub == 0.
                && r.lb == f64::NEG_INFINITY
                && r.coeffs.len() == 2
                && r.coeffs.contains(&(heads[2].0, 1.))
                && r.coeffs.contains(&(selector, -1.))
        })
        .unwrap();
    weighted_row(
        upper,
        Q::new((-1649).into(), 3376.into()),
        &mut combined,
        &mut rhs,
    );
    let lower = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.lb == 0.
                && r.ub == f64::INFINITY
                && r.coeffs.len() == 2
                && r.coeffs.contains(&(heads[1].0, 1.))
                && r.coeffs.contains(&(31, -1.))
        })
        .unwrap();
    weighted_row(
        lower,
        Q::new(1649.into(), 6752.into()),
        &mut combined,
        &mut rhs,
    );
    let affine = &original.problem.rows()[1];
    assert_eq!(affine.lb, affine.ub);
    let a = affine.coeffs.iter().find(|(j, _)| *j == 31).unwrap().1;
    let weight = -combined[31].clone() / q(a);
    weighted_row(affine, weight, &mut combined, &mut rhs);
    assert!(combined[31].is_zero());
    let mut objective = vec![Q::zero(); parts.num_cols];
    for j in 0..18 {
        objective[heads[j].0] = q(f.w[1][j]) - q(f.w[1][18 + j]);
    }
    let mut rigorous = q(f.b[1][0]) - q(f.b[1][1]) + rhs;
    for (j, (want, got)) in objective.iter().zip(&combined).enumerate() {
        let residual = want - got;
        if residual.is_zero() {
            continue;
        }
        let col = &parts.problem.cols()[j];
        assert!(
            col.lb.is_finite() && col.ub.is_finite(),
            "unpriced column{j}"
        );
        rigorous += (&residual * q(col.lb)).min(&residual * q(col.ub));
    }
    rigorous
}

#[test]
fn cell_default_conditional_route_counts_table_lp_separately_from_original_lp() {
    let f = fixture();
    let p = f.source();
    let cuts =
        propose_root_mip_cuts_with_conditional_mass_portfolio_attempt(&p, end(), Duration::ZERO)
            .cuts
            .unwrap();
    let owner = cuts.signed.as_ref().unwrap();
    let k = owner
        .plan
        .components
        .iter()
        .map(|c| c.negative.len())
        .max()
        .unwrap();
    let initial = owner.stats();
    assert!(initial.component_rows.phase.cube.cell.admitted);
    assert!(initial.component_rows.phase.cube.cell.ray.admitted);
    assert!(initial.component_rows.phase.cube.cell.lp.admitted);
    let mut parts = f.original();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let stats = report.signed.as_ref().unwrap();
    let cell = &stats.component_rows.phase.cube.cell;
    assert!(cell.attempted && cell.admitted);
    assert_eq!(cell.collected_masks, 1 << k);
    assert!(cell.improved_masks > 0);
    assert_eq!(cell.appended_rows, 1);
    assert!(cell.ray.admitted);
    assert_eq!(cell.ray.masks, 1 << k);
    assert!(cell.lp.admitted);
    assert_eq!(cell.lp.calls, 1);
    assert!(!cell.lp.activity_from_root);
    assert_eq!(cell.lp.checked_masks, 1 << k);
    assert!(cell.lp.pivots <= cell.lp.pivot_limit);
    assert_eq!(stats.source_replays, 1);
    assert_eq!(report.proposal_lp_calls, 1);
    assert_eq!(cuts.native_funding_report().funded_calls, 1);
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    let old = propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt(
        &p,
        end(),
        Duration::ZERO,
    )
    .cuts
    .unwrap();
    let mut baseline = f.original();
    old.append_to(&mut baseline, &f.encoding(), end()).unwrap();
    assert_eq!(parts.problem.cols(), baseline.problem.cols());
    for row in baseline.problem.rows() {
        assert!(
            parts.problem.rows().contains(row),
            "lost previous row {row:?}"
        );
    }
}

#[test]
fn conditional_portfolio_automatic_loser_closes_original_lp_and_ay() {
    let f = fixture();
    let p = f.source();
    let old = propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt(
        &p,
        end(),
        Duration::ZERO,
    )
    .cuts
    .unwrap();
    let cuts = candidate(&p);
    let original = f.original();
    let encoding = f.encoding();
    let mut baseline = f.original();
    old.append_to(&mut baseline, &encoding, end()).unwrap();
    let values = fake(&f, &original);
    rows_hold(&baseline, &values, None, usize::MAX);
    let mut parts = f.original();
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    let stats = cuts.mass_portfolio_report();
    let signed = report.signed.as_ref().unwrap();
    eprintln!("automatic conditional portfolio: {stats:?}; signed={signed:?}");
    assert_eq!(signed.component_rows.phase.cube.appended_rows, 0);
    assert!(stats.phase_appended_rows >= 1);
    assert_eq!(stats.appended_rows - stats.phase_appended_rows, 11);
    assert_eq!(report.proposal_lp_calls, 1);
    assert_eq!(signed.source_replays, 1);
    assert_eq!(
        signed.checked_masks,
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
    assert_eq!(
        &parts.problem.rows()[..original.problem.num_rows()],
        original.problem.rows()
    );
    // Every old row survives, including the plain losing allocation.
    for row in baseline.problem.rows() {
        assert!(parts.problem.rows().contains(row), "lost old row {row:?}");
    }
    let tail = parts.problem.num_rows() - stats.phase_appended_rows;
    rows_hold(&parts, &values, None, tail);
    assert!(parts.problem.rows()[tail..].iter().any(|r| {
        let lhs: Q = r.coeffs.iter().map(|(j, a)| q(*a) * &values[*j]).sum();
        lhs < q(r.lb)
    }));
    let rigorous = rational_certificate(&f, &p, &parts, &original, &stats);
    eprintln!("exact stored-row lower certificate: {rigorous}");
    assert!(rigorous > q(1. / 256.));
    assert!(rigorous >= Q::new(2499.into(), 432128.into()) - q(2f64.powi(-30)));
    let before = lp_floor(&baseline, encoding.unsafe_row.0, usize::MAX);
    let after = lp_floor(&parts, encoding.unsafe_row.0, usize::MAX);
    eprintln!("actual original LP lower: before={before} after={after}");
    assert!(before <= 0.);
    assert!(after > 1. / 256.);
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

#[test]
fn conditional_portfolio_single_false_poll_aborts_original_append() {
    let f = fixture();
    let p = f.source();
    let cuts = candidate(&p);
    let mut parts = f.original();
    let original = parts.problem.clone();
    let mut stopped = false;
    let result = cuts.append_with_poll(&mut parts, &f.encoding(), end(), &mut || {
        if !stopped && cuts.mass_portfolio_report().export_attempts == 1 {
            stopped = true;
            false
        } else {
            true
        }
    });
    assert!(stopped && result.is_none());
    assert_eq!(parts.problem.rows(), original.rows());
    assert_eq!(parts.problem.cols(), original.cols());
    let stats = cuts.mass_portfolio_report();
    assert_eq!(stats.phase_appended_rows, 0);
    assert_eq!(stats.appended_rows, 0);
    assert_eq!(stats.reserved_ir_rows, 0);
}

fn graph_point(f: &Fixture, parts: &MipParts, x: Q, zero_mask: usize) -> Vec<Q> {
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    let mut values = vec![Q::zero(); parts.num_cols];
    values[parts.input_vars[29].0] = x.clone();
    for j in 0..18 {
        let z = q(f.b[0][j]) + q(f.w[0][30 * j + 29]) * &x;
        values[30 + j] = z.clone();
        values[heads[j].0] = z.clone().max(Q::zero());
        let bounds = f.bounds[0][j];
        if bounds.lower() < 0. && bounds.upper() > 0. {
            let selector = crate::ir::Col(heads[j].0 + 1);
            assert!(original.binary_vars.contains(&selector));
            values[selector.0] = q(if z.is_zero() {
                ((zero_mask >> j) & 1) as f64
            } else {
                usize::from(z > Q::zero()) as f64
            });
        }
    }
    for output in 0..2 {
        values[parts.output_vars[output].0] = q(f.b[1][output])
            + (0..18)
                .map(|j| q(f.w[1][18 * output + j]) * &values[heads[j].0])
                .sum::<Q>();
    }
    values
}

#[test]
fn conditional_portfolio_preserves_original_graph_unsafe_witness_and_closed_tie() {
    for bias in [5. - 1. / 512., 0., -1. / 512.] {
        let mut f = fixture();
        f.b[1][0] = bias;
        let p = f.source();
        let cuts = propose_root_mip_cuts_with_conditional_mass_portfolio_attempt(
            &p,
            end(),
            Duration::ZERO,
        )
        .cuts
        .unwrap();
        let mut parts = f.original();
        let encoding = f.encoding();
        cuts.append_to(&mut parts, &encoding, end()).unwrap();
        assert!(cuts.mass_portfolio_report().phase_appended_rows > 0);
        for x in [-1., -1. / 64., 0., 1.] {
            for mask in [0usize, (1 << 18) - 1, 0x15555, 0x2aaaa] {
                let point = graph_point(&f, &parts, q(x), mask);
                let skip = if bias <= 0. && x == 0. {
                    None
                } else {
                    Some(encoding.unsafe_row.0)
                };
                rows_hold(&parts, &point, skip, usize::MAX);
                if skip.is_none() {
                    assert!(point[parts.output_vars[0].0] <= point[parts.output_vars[1].0]);
                    assert!(cuts.best_signed_root_proof(end()).is_none());
                }
            }
        }
    }
}
