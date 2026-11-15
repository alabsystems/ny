// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! The new CLI-selected exclusive route retains a useful unstepped seed.
//! These actual-default controls exercise SeedSingle, not tau-branch discovery.
use super::super::test_fixture::{end, lp, q, rows_hold, solve, Fixture};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

fn pressure(bias: f64) -> Fixture {
    // True margin = bias +54*(sum_i R(x_i)-R(sum_i x_i))
    // plus nonnegative unrelated heads. Its exact minimum is bias at zero.
    let (d, h) = (30, 128);
    let mut w = vec![0.; d * h];
    for j in 0..60 {
        let axis = if j < 6 { j } else { 6 + (j - 6) % 24 };
        w[j * d + axis] = 1.;
    }
    for j in 60..114 {
        for i in 0..6 {
            w[j * d + i] = 1.;
        }
    }
    let mut c = vec![0.; 2 * h];
    c[..6].fill(54.);
    c[6..60].fill(1. / 1024.);
    c[60..114].fill(-1.);
    let mut bounds = vec![Bound::new(0., 0.); h];
    bounds[..60].fill(Bound::new(-1., 1.));
    bounds[60..114].fill(Bound::new(-6., 6.));
    Fixture {
        w: vec![w, c],
        b: vec![vec![0.; h], vec![bias, 0.]],
        dims: vec![d, h, 2],
        input: vec![Bound::new(-1., 1.); d],
        bounds: vec![bounds],
    }
}

fn graph_point(f: &Fixture, parts: &MipParts, input: &[Q], zero_active: bool) -> Vec<Q> {
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    let mut values = vec![Q::zero(); parts.num_cols];
    for (i, value) in input.iter().enumerate() {
        values[parts.input_vars[i].0] = value.clone();
    }
    let mut binary = 0;
    for j in 0..f.dims[1] {
        let z = q(f.b[0][j])
            + (0..f.dims[0])
                .map(|i| q(f.w[0][j * f.dims[0] + i]) * &input[i])
                .sum::<Q>();
        values[f.dims[0] + j] = z.clone();
        values[heads[j].0] = z.clone().max(Q::zero());
        let bound = f.bounds[0][j];
        if bound.lower() < 0. && bound.upper() > 0. {
            let active = if z.is_zero() {
                zero_active
            } else {
                z > Q::zero()
            };
            values[original.binary_vars[binary].0] = q(usize::from(active) as f64);
            binary += 1;
        }
    }
    for o in 0..2 {
        values[parts.output_vars[o].0] = q(f.b[1][o])
            + (0..f.dims[1])
                .map(|j| q(f.w[1][o * f.dims[1] + j]) * &values[heads[j].0])
                .sum::<Q>();
    }
    values
}

fn selected_cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    // This is the same public entry selected by CLI strengthen_with_signed_refinement.
    // No manual mode bit, partition, Point, lambda or nominal target is installed.
    let attempt = propose_root_mip_cuts_with_signed_branch_tau_attempt(p, end(), Duration::ZERO);
    assert!(
        attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked,
        "capture={:?}",
        attempt.primal
    );
    let cuts = attempt
        .cuts
        .expect("the real default route retains its owner");
    assert!(
        cuts.group_allocation_mode && cuts.branch_tau_mode,
        "the CLI-selected exclusive branch mode must be carried"
    );
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert!(
        cuts.signed.as_ref().is_some_and(|row| row.lower < 0.),
        "the old signed scalar must remain OPEN in this discovery control"
    );
    cuts
}

fn appended(f: &Fixture, p: &HullMarginProblem) -> (MipParts, HullMipCutReport, usize, usize) {
    let cuts = selected_cuts(p);
    let capture = cuts.lp_primal_report();
    assert_eq!((capture.positive_heads, capture.negative_heads), (60, 54));
    assert_eq!(capture.work_units, 153_664);
    let components = cuts.signed.as_ref().unwrap().plan.components.len();
    assert_eq!(components, 68);
    // Compute the full unpaid quote without changing the real default route.
    // Actual append below must pay max(single-gradient+aliases, branch-tau),
    // even though this fixture's useful seed chooses only the canonical tail.
    let base = dense::quote_with_aliases(p, components, capture.work_units, 60, 54, 7).unwrap();
    let old_incremental = base.incremental_work;
    let old_single_tail = base.step_work + 64 * 30 * 5 + base.alias_work;
    let single_tail = old_single_tail + 64 * 7; // Common scaled-lambda eligibility.
    let planning = Planning::new(p).unwrap();
    let paid = planning
        .quote_base(p, components, capture.work_units, base)
        .unwrap();
    assert_eq!(
        (paid.nominal, paid.optimizer, paid.checker),
        (33_216, 84_160, 35_840)
    );
    assert_eq!((paid.single, paid.branch), (single_tail, 153_216));
    assert_eq!(
        paid.incremental_work(),
        old_incremental - old_single_tail + single_tail.max(153_216)
    );
    assert_eq!(paid.total_work(), 660_240);
    let next = dense::quote_with_aliases(p, components, capture.work_units, 60, 54, 8).unwrap();
    assert!(planning
        .quote_base(p, components, capture.work_units, next)
        .is_none());
    let mut parts = f.parts();
    let before = parts.problem.num_rows();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    assert_eq!(parts.problem.num_rows() - before, report.cuts);
    let signed = report.signed.as_ref().unwrap();
    let dom = &signed.component_rows.domination;
    assert!(
        dom.admitted && dom.dense && dom.group_allocation && dom.branch_tau_mode,
        "route={dom:?}"
    );
    assert_eq!(dom.funded_positive_limit, 7);
    assert_eq!(dom.exclusive_seed_useful, Some(true));
    assert!(!dom.gradient_skipped);
    assert_eq!(
        (dom.funded_targets, dom.advice_attempts, dom.checker_slots),
        (1, 1, 1)
    );
    assert_eq!(
        (
            dom.support_attempts,
            dom.completed_supports,
            dom.export_attempts
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (
            dom.appended_rows,
            dom.funded_alias_heads,
            dom.checked_alias_rows,
            dom.appended_alias_rows
        ),
        (54, 54, 54, 54)
    );
    assert_eq!(dom.prepared_head_terms, 378); // Six positive partners plus each original target.
                                              // The immutable maximum-tail fee was paid before route selection. A
                                              // reserved optimizer/checker fee is not completed branch work or a row.
    assert_eq!(
        (dom.tail_single_work, dom.tail_branch_work),
        (paid.single, paid.branch)
    );
    assert_eq!(
        (
            dom.group_nominal_work,
            dom.group_checker_tail,
            dom.branch_tau_optimizer_work
        ),
        (paid.nominal, paid.checker, paid.optimizer)
    );
    assert_eq!(dom.work_units, 506_576);
    assert_eq!(dom.total_optional_work_units, paid.total_work());
    assert_eq!(
        (
            dom.group_nominal_attempts,
            dom.group_nominal_masks,
            dom.group_checked_masks
        ),
        (0, 0, 0)
    );
    assert_eq!(
        (
            dom.branch_tau_attempts,
            dom.branch_tau_completed,
            dom.branch_tau_checked_masks
        ),
        (0, 0, 0)
    );
    assert!(dom.branch_tau_values.is_none() && dom.branch_tau_mask_prices.is_none());
    assert!(!dom.branch_tau_selected);
    assert_eq!(dom.capture_work_units, capture.work_units);
    assert_eq!(
        dom.total_optional_work_units,
        capture.work_units + dom.work_units
    );
    assert!(dom.total_optional_work_units <= resources::MAX_DOMINATION_WORK);
    assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
    assert_eq!((report.proposal_lp_calls, signed.source_replays), (1, 1));
    let stop = parts.problem.num_rows() - signed.component_rows.joint.appended_rows;
    let start = stop - dom.appended_rows;
    assert!(start >= before && stop - start == 54);
    for value in [-1.0_f64, 0., 1.] {
        for active in [false, true] {
            let point = graph_point(f, &parts, &vec![q(value); 30], active);
            // General graph points need not satisfy the unsafe property.
            rows_hold(&parts.problem, &point, Some(f.encoding().unsafe_row.0));
        }
    }
    (parts, report, start, stop)
}

#[test]
fn branch_default_cli_selected_seed_single_keeps_max_tail_aliases_and_actual_safe_certificate() {
    let f = pressure(0.25);
    let p = f.source();
    let (parts, _report, start, stop) = appended(&f, &p);
    let original = f.parts();
    let property = f.encoding().unsafe_row.0;
    let before = lp(&original.problem, &original, property);
    assert!(
        (before - (0.25 - 162.)).abs() < 1e-6,
        "original Big-M={before}"
    );
    let mut ir = original.problem.clone();
    let row = &parts.problem.rows()[start];
    ir.add_row(
        row.lb,
        row.ub,
        row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
    );
    let one = lp(&ir, &original, property);
    assert!(
        (one - (0.25 - 159.)).abs() < 1e-6,
        "one canonical row={one}"
    );
    assert!(one > before + 2.999);
    for row in &parts.problem.rows()[start + 1..stop] {
        ir.add_row(
            row.lb,
            row.ub,
            row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
        );
    }
    let all = lp(&ir, &original, property);
    assert!(
        (all - 0.25).abs() < 1e-6 && all > 0.249,
        "all54 aliases={all}"
    );
    // The original full graph plus actual selected rows reaches the real AY
    // proof/checker path, rather than inferring safety from an LP telemetry value.
    let result = solve(parts);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

fn actual_sat(bias: f64) {
    let f = pressure(bias);
    let p = f.source();
    let (parts, _report, _, _) = appended(&f, &p);
    for active in [false, true] {
        let point = graph_point(&f, &parts, &vec![Q::zero(); 30], active);
        rows_hold(&parts.problem, &point, None);
        assert_eq!(point[parts.output_vars[0].0], q(bias));
        assert!(point[parts.output_vars[0].0] <= Q::zero());
    }
    match solve(parts) {
        crate::MipResult::Sat { input_values, .. } => {
            let input: Vec<_> = input_values.iter().map(|&x| q(x)).collect();
            let original = f.original();
            let point = graph_point(&f, &original, &input, false);
            rows_hold(&original.problem, &point, None);
            assert!(point[original.output_vars[0].0] <= Q::zero());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn branch_default_cli_selected_unsafe_graph_keeps_max_tail_and_original_sat_witness() {
    actual_sat(-0.25);
}

#[test]
fn branch_default_cli_selected_zero_tie_keeps_max_tail_and_original_sat_witness() {
    actual_sat(0.);
}
