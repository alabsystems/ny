// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual default LP/IR discovery and additive bounded-owner controls.
use super::super::test_fixture::{self as fixture, end, lp, q, rows_hold, solve, Fixture};
use super::*;
use crate::single_hidden_hull::primal_advice;
use num_rational::BigRational as Q;
use num_traits::Zero;

fn pressure(bias: f64) -> Fixture {
    // Six useful independent positive heads,54 unrelated positive heads,
    //54 original negative copies,14 inert heads. No asset-specific selectors.
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
fn actual_cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let answer = propose_root_mip_cuts_with_signed_refinement_attempt(p, end(), Duration::ZERO);
    assert!(
        answer.primal.same_call && answer.primal.decoded && answer.primal.banked,
        "actual capture={:?}",
        answer.primal
    );
    let cuts = answer
        .cuts
        .expect("actual default attempt must retain an owner");
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert!(
        cuts.signed.as_ref().is_some_and(|r| r.lower < 0.),
        "the generic pressure control requires an OPEN old signed root"
    );
    cuts
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
fn first(row: &SignedMajorant, cuts: &HullRootMipCuts<'_>) {
    assert!(row
        .replay_inner(cuts, row.authority(), &mut || true, false)
        .is_some());
    let c = row.budget.state.lock().unwrap();
    assert_eq!((c.replay_attempts, c.source_replays), (1, 1));
}
fn manual_old_plan(p: &HullMarginProblem) -> Plan {
    let negatives: Vec<_> = (0..p.hidden_dim).filter(|&j| p.c[j] < 0.).collect();
    let components = negatives
        .chunks(8)
        .enumerate()
        .map(|(i, n)| Component {
            negative: n.to_vec(),
            positive: if i == 0 { (0..6).collect() } else { vec![] },
            positive_units: if i == 0 { vec![2; 6] } else { vec![] },
            alpha: if i == 0 {
                vec![0.5; (1usize << n.len()) * 6]
            } else {
                vec![]
            },
            g: vec![0.; p.input_dim],
        })
        .collect();
    Plan {
        components,
        leftover: (6..60).collect(),
    }
}

#[test]
fn dense_advice_real_quote_adapts_adjacent_limits_and_pays_low_dimension_duplicate_scans() {
    let p = pressure(0.25).source();
    let capture = primal_advice::quote(
        &p,
        primal_advice::RoleCounts {
            positive: 60,
            negative: 54,
        },
    )
    .unwrap();
    let capture_work = capture.debt().0;
    assert_eq!(capture_work, 153_664);
    // Cached seed + both merge orders + one projected step are fully paid.
    let paid = quote_with_aliases(&p, 16, capture_work, 60, 0, 26).unwrap();
    assert_eq!(paid.total_work, 656_944);
    assert_eq!(paid.incremental_work + capture_work, paid.total_work);
    assert!(quote_with_aliases(&p, 16, capture_work, 60, 0, 27).is_none());
    assert!(paid.total_work <= resources::MAX_DOMINATION_WORK);
    assert_eq!(paid.positive_limit, 26);
    assert!(paid.bytes <= MAX_BYTES);
    // All 54 unstable and 14 rounded-negative dead heads may be separate
    // components in the ORIGINAL planner. Keep the same work cap and adapt q.
    let singleton = quote_with_aliases(&p, 68, capture_work, 60, 0, 14).unwrap();
    assert_eq!(singleton.total_work, 661_776);
    assert_eq!(singleton.positive_limit, 14);
    assert!(quote_with_aliases(&p, 68, capture_work, 60, 0, 15).is_none());
    let aliases = quote_with_aliases(&p, 68, capture_work, 60, 54, 10).unwrap();
    assert_eq!(
        (aliases.total_work, aliases.incremental_work, aliases.bytes),
        (664_064, 510_400, 124_528)
    );
    assert_eq!((aliases.alias_work, aliases.alias_bytes), (48_976, 46_432));
    assert!(quote_with_aliases(&p, 68, capture_work, 60, 54, 11).is_none());
    // A second engineering shape varies actual-role counts without choosing
    // an asset or querying a model; both adjacent widths obey the same cap.
    let shifted_capture = primal_advice::quote(
        &p,
        primal_advice::RoleCounts {
            positive: 62,
            negative: 52,
        },
    )
    .unwrap()
    .debt()
    .0;
    assert_eq!(shifted_capture, 148_864);
    let shifted = quote_with_aliases(&p, 16, shifted_capture, 62, 52, 22).unwrap();
    assert_eq!(
        (shifted.total_work, shifted.incremental_work),
        (660_416, 511_552)
    );
    assert!(quote_with_aliases(&p, 16, shifted_capture, 62, 52, 23).is_none());
    assert!(quote_with_aliases(&p, p.hidden_dim + 1, capture_work, 60, 0, 1).is_none());
    let mut narrow = p.clone();
    narrow.input_dim = 1;
    narrow.w1 = vec![1.; narrow.hidden_dim];
    narrow.lower = vec![-1.];
    narrow.upper = vec![1.];
    let capture = primal_advice::quote(
        &narrow,
        primal_advice::RoleCounts {
            positive: 60,
            negative: 54,
        },
    )
    .unwrap();
    assert_eq!(capture.debt().0, 26_064);
    // At D1 both component counts fit the arithmetic cap, so this checks the
    // shape limit itself rather than accidentally refusing on total work.
    assert_eq!(
        quote_with_aliases(&narrow, 128, capture.debt().0, 60, 0, 1)
            .unwrap()
            .total_work,
        598_448
    );
    assert!(quote_with_aliases(&narrow, 129, capture.debt().0, 60, 0, 1).is_none());
    let dense = quote_with_aliases(&narrow, 16, capture.debt().0, 60, 0, 32).unwrap();
    // Two complete uniqueness checks each cost4*q*(q-1), even at D1.
    assert_eq!(8 * 32 * (32 - 1), 7_936);
    assert_eq!(dense.incremental_work, 118_336);
    assert_eq!(dense.total_work, 144_400);
    assert_eq!(
        quote_with_aliases(&narrow, 16, capture.debt().0, 60, 0, 33)
            .unwrap()
            .positive_limit,
        32
    );
    assert!(quote_with_aliases(&p, 16, resources::MAX_DOMINATION_WORK, 60, 0, 1).is_none());
}

#[test]
fn dense_advice_actual_default_more_than_four_partners_strengthen_original_ir_and_keep_ay_safety() {
    for bias in [0.25, 0., -0.25] {
        let f = pressure(bias);
        let p = f.source();
        let cuts = actual_cuts(&p);
        let capture = cuts.lp_primal_report();
        assert_eq!((capture.positive_heads, capture.negative_heads), (60, 54));
        assert_eq!(capture.work_units, 153_664);
        let mut parts = f.parts();
        let original_rows = parts.problem.num_rows();
        let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        let signed = report.signed.as_ref().unwrap();
        let dom = &signed.component_rows.domination;
        assert!(
            dom.dense && dom.admitted,
            "actual default domination={dom:?}"
        );
        assert!(dom.funded_positive_limit > 4 && dom.funded_positive_limit <= 32);
        assert_eq!(
            (
                dom.funded_targets,
                dom.support_attempts,
                dom.completed_supports,
                dom.appended_rows
            ),
            (1, 1, 1, 54)
        );
        assert_eq!(
            (
                dom.funded_alias_heads,
                dom.checked_alias_rows,
                dom.appended_alias_rows
            ),
            (54, 54, 54)
        );
        assert_eq!(dom.prepared_head_terms, 378); //54 copies of six partners plus target.
        assert_eq!(dom.capture_work_units, capture.work_units);
        assert_eq!(
            dom.total_optional_work_units,
            capture.work_units + dom.work_units
        );
        assert!(dom.total_optional_work_units <= resources::MAX_DOMINATION_WORK);
        assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
        assert_eq!((report.proposal_lp_calls, signed.source_replays), (1, 1));
        let dom_end = parts.problem.num_rows() - signed.component_rows.joint.appended_rows;
        let dom_start = dom_end - dom.appended_rows;
        assert!(dom_start >= original_rows);
        for value in [-1., 0., 1.] {
            for tie in [false, true] {
                let point = graph_point(&f, &parts, &vec![q(value); 30], tie);
                rows_hold(&parts.problem, &point, Some(f.encoding().unsafe_row.0));
            }
        }
        if bias <= 0. {
            for tie in [false, true] {
                let point = graph_point(&f, &parts, &vec![Q::zero(); 30], tie);
                rows_hold(&parts.problem, &point, None);
                assert_eq!(point[parts.output_vars[0].0], q(bias));
            }
        }
        if bias > 0. {
            // This comparison is against the ORIGINAL individual Big-M IR,
            // not a claim that every later old component/phase row misses it.
            let original = f.parts();
            let unsafe_row = f.encoding().unsafe_row.0;
            let before = lp(&original.problem, &original, unsafe_row);
            let mut strengthened = original.problem.clone();
            for row in &parts.problem.rows()[dom_start..=dom_start] {
                strengthened.add_row(
                    row.lb,
                    row.ub,
                    row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
                );
            }
            let after = lp(&strengthened, &original, unsafe_row);
            assert!(
                (before - (bias - 162.)).abs() < 1e-6,
                "original LP={before}"
            );
            assert!(
                (after - (bias - 159.)).abs() < 1e-6,
                "one dense row LP={after}"
            );
            assert!(after > before + 2.999);
            // Exact original-affine reuse binds EVERY negative copy. The
            // single canonical LP comparison above remains independently pinned.
            for row in &parts.problem.rows()[dom_start + 1..dom_end] {
                strengthened.add_row(
                    row.lb,
                    row.ub,
                    row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
                );
            }
            let all_aliases = lp(&strengthened, &original, unsafe_row);
            assert!(
                all_aliases > 0.249 && (all_aliases - bias).abs() < 1e-6,
                "all exact-alias rows LP={all_aliases}"
            );
            let result = solve(parts);
            assert!(
                matches!(result, crate::MipResult::Unsat { certified: true }),
                "{result:?}"
            );
        } else {
            let result = solve(parts);
            match result {
                crate::MipResult::Sat { input_values, .. } => {
                    let input: Vec<_> = input_values.iter().map(|&x| q(x)).collect();
                    let original = f.original();
                    let replay = graph_point(&f, &original, &input, false);
                    rows_hold(&original.problem, &replay, None);
                    assert!(replay[original.output_vars[0].0] <= Q::zero());
                }
                other => panic!("{other:?}"),
            }
        }
    }
}

#[test]
fn dense_advice_capture_debt_and_payload_bank_exactly_once_before_original_search() {
    let f = Fixture::sparse(0.25);
    let p = f.source();
    let mut attempt = primal_capture::Attempt::new(&p, end());
    let mut cuts = propose_root_mip_cuts_inner(&p, end(), Some(&mut attempt)).unwrap();
    let capture = cuts.lp_primal_report();
    let plain = propose_root_mip_cuts(&p, end()).unwrap();
    let old = Budget::new(&plain, end(), Duration::ZERO).unwrap();
    assert_eq!(old.state.lock().unwrap().work, 0);
    let bank = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    assert_eq!(bank.state.lock().unwrap().work, capture.work_units);
    assert_eq!(bank.admitted_bytes, old.admitted_bytes + capture.peak_bytes);
    assert!(cuts.lp_primal_report().banked);
    assert!(Budget::new(&cuts, end(), Duration::ZERO).is_none());
    assert_eq!(bank.state.lock().unwrap().work, capture.work_units);
    assert_eq!(cuts.lp_primal_report().bank_attempts, 2);
    cuts.primal = None;
    assert!(Budget::new(&cuts, end(), Duration::ZERO).is_none());
    assert_eq!(bank.state.lock().unwrap().work, capture.work_units);
}

#[test]
fn dense_advice_old_sparse_ticket_cannot_pay_for_five_partners_or_a_third_publication() {
    let f = pressure(0.25);
    let p = f.source();
    let cuts = fixture::cuts_with_plan(&p, manual_old_plan(&p));
    let row = cuts.signed.as_ref().unwrap();
    first(row, &cuts);
    let ticket = resources::admit(row, &cuts, row.budget.search_end).unwrap();
    assert_eq!(ticket.support_limit(), 4);
    assert!(ticket.dense_quote().is_none());
    let before = row.stats().work_units;
    let proposal = kernel::Proposal {
        head: 60,
        positive: (0..5).collect(),
        lambda: vec![1.; 5],
        v: vec![0.; 30],
    };
    assert!(kernel::upper(&p, &proposal, row.budget.search_end, &mut || true).is_some());
    assert!(kernel::check(row, &cuts, &ticket, proposal, ticket.end, &mut || true).is_none());
    let stats = row.stats();
    assert_eq!(stats.component_rows.domination.support_attempts, 1);
    assert_eq!(stats.component_rows.domination.completed_supports, 0);
    assert_eq!(stats.work_units, before);
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_some());
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_none());
    assert_eq!(row.stats().source_replays, 2);
    assert!(row.stats().work_units <= MAX_WORK);
}

#[test]
fn dense_advice_immutable_captured_debt_survives_report_corruption_and_quota_refusal() {
    let f = pressure(0.25);
    let p = f.source();
    let cuts = actual_cuts(&p);
    let row = cuts.signed.as_ref().unwrap();
    first(row, &cuts);
    let work = cuts.lp_primal_report().work_units;
    {
        let mut report = cuts.primal_report.as_ref().unwrap().lock().unwrap();
        report.work_units = 0;
    }
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let mut ticket =
        resources::admit_point(row, &cuts, row.budget.search_end, Some(&account)).unwrap();
    assert!(ticket.dense_quote().is_some());
    assert_eq!(
        row.stats().component_rows.domination.capture_work_units,
        work
    );
    assert!(
        row.stats()
            .component_rows
            .domination
            .total_optional_work_units
            <= resources::MAX_DOMINATION_WORK
    );
    let accepted_end = ticket.end;
    ticket.end += Duration::from_millis(1);
    assert!(resources::support_attempt(row, &ticket, ticket.end).is_none());
    assert_eq!(row.stats().component_rows.domination.support_attempts, 0);
    ticket.end = accepted_end;
    let count = ticket.support_limit() + 1;
    let too_wide = kernel::Proposal {
        head: 60,
        positive: (0..count).collect(),
        lambda: vec![1.; count],
        v: vec![0.; 30],
    };
    assert!(kernel::check(row, &cuts, &ticket, too_wide, accepted_end, &mut || true).is_none());
    assert!(resources::support_attempt(row, &ticket, accepted_end).is_none());
    assert_eq!(row.stats().component_rows.domination.support_attempts, 1);
    assert_eq!(row.stats().component_rows.domination.completed_supports, 0);
}

#[test]
fn dense_advice_final_support_poll_refusal_keeps_paid_work_and_both_replay_limits() {
    let f = pressure(0.25);
    let p = f.source();
    let cuts = actual_cuts(&p);
    let row = cuts.signed.as_ref().unwrap();
    first(row, &cuts);
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let ticket = resources::admit_point(row, &cuts, row.budget.search_end, Some(&account)).unwrap();
    let point = account.point(&p, ticket.end, &row.budget.state).unwrap();
    resources::begin_advice(row, &ticket).unwrap();
    let proposal = advice(
        row,
        &cuts,
        point,
        ticket.dense_quote().unwrap(),
        ticket.end,
        &mut || true,
    )
    .filter(|advice| advice.useful)
    .map(|advice| advice.single)
    .unwrap();
    assert!(proposal.positive.len() > 4);
    let mut polls = 0;
    kernel::upper(&p, &proposal, ticket.end, &mut || {
        polls += 1;
        true
    })
    .unwrap();
    assert!(polls > 2);
    let before = row.stats().work_units;
    let mut seen = 0;
    assert!(
        kernel::check(row, &cuts, &ticket, proposal, ticket.end, &mut || {
            seen += 1;
            seen < polls
        })
        .is_none()
    );
    assert_eq!(seen, polls);
    assert_eq!(row.stats().work_units, before);
    assert_eq!(row.stats().component_rows.domination.completed_supports, 0);
    assert!(resources::begin_advice(row, &ticket).is_none());
    drop(account);
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_some());
    assert!(row
        .replay_inner(&cuts, row.authority(), &mut || true, false)
        .is_none());
    assert_eq!(row.stats().source_replays, 2);
    assert!(row.stats().work_units <= MAX_WORK);
}

#[test]
fn dense_advice_exact_all_four_partner_projections_admit_a_point_excluded_by_the_default_row() {
    let f = pressure(0.25);
    let original = f.parts();
    let heads = binding::head_columns(&f.encoding(), &f.original()).unwrap();
    let zero = vec![Q::zero(); 30];
    let mut fake = graph_point(&f, &original, &zero, false);
    for j in 60..114 {
        fake[heads[j].0] = q(1.);
    }
    for col in &f.original().binary_vars {
        fake[col.0] = q(0.5);
    }
    fake[original.output_vars[0].0] = q(0.25) - q(54.);
    rows_hold(&original.problem, &fake, None);
    let mut projections = 0;
    for mask in 0u32..64 {
        if mask.count_ones() > 4 {
            continue;
        }
        let uncovered: Vec<_> = (0..6).filter(|&i| mask & (1u32 << i) == 0).collect();
        assert!(uncovered.len() >= 2);
        let mut positive = zero.clone();
        positive[uncovered[0]] = q(1.);
        positive[uncovered[1]] = q(1.);
        let negative: Vec<_> = positive.iter().map(|x| -x).collect();
        let plus = graph_point(&f, &original, &positive, false);
        let minus = graph_point(&f, &original, &negative, true);
        rows_hold(&original.problem, &plus, Some(f.encoding().unsafe_row.0));
        rows_hold(&original.problem, &minus, Some(f.encoding().unsafe_row.0));
        for col in &original.input_vars {
            assert_eq!((&plus[col.0] + &minus[col.0]) / q(2.), Q::zero());
        }
        for j in 60..114 {
            assert_eq!((&plus[heads[j].0] + &minus[heads[j].0]) / q(2.), q(1.));
        }
        for j in 0..60 {
            if j >= 6 || mask & (1u32 << j) != 0 {
                assert_eq!((&plus[heads[j].0] + &minus[heads[j].0]) / q(2.), Q::zero());
            }
        }
        projections += 1;
    }
    assert_eq!(projections, 57);
    // For ANY <=4 selected positives (and arbitrary lambda/v), these two
    // original graph atoms have the fake point's input/selected-positive/ALL
    // negative projection. Thus every valid such affine domination row admits
    // it. This does not assert membership in all other old row families.
    let p = f.source();
    let cuts = actual_cuts(&p);
    let mut parts = f.parts();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let signed = report.signed.as_ref().unwrap();
    let dom = &signed.component_rows.domination;
    assert!(dom.dense && dom.prepared_head_terms > 5);
    assert_eq!((dom.appended_rows, dom.appended_alias_rows), (54, 54));
    let last = parts.problem.num_rows() - signed.component_rows.joint.appended_rows;
    let row = &parts.problem.rows()[last - 1];
    let lhs: Q = row.coeffs.iter().map(|(j, a)| q(*a) * &fake[*j]).sum();
    assert!((row.lb.is_finite() && lhs < q(row.lb)) || (row.ub.is_finite() && lhs > q(row.ub)));
}
