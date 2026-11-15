// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Adversarial controls of optional table authority and retention.
use super::super::default_tests::cube_tests::{candidate, fixture};
use super::super::tests::{loser, prepared, source};
use super::*;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn losing_cube_tamper_and_numeric_refusal_preserve_every_prior_row() {
    let f = fixture(79. / 64.);
    let p = f.source();
    let old = candidate(&p, false);
    let mut baseline = f.original();
    old.append_to(&mut baseline, &f.encoding(), end()).unwrap();
    for mode in 0..4 {
        let cuts = candidate(&p, true);
        let debt = {
            let mut c = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
            let family = c.portfolio.recipe.as_mut().unwrap();
            let extra = family.conditional.as_mut().unwrap().cube_for_test();
            match mode {
                0 => extra.tables[0][0] = f64::NAN,
                1 => extra.seal[0] ^= 1,
                2 => extra.tables[0].swap(0, 1),
                _ => {
                    let lower = family.planes[0].lower * 0.5;
                    extra.tables[0].fill(f64::MAX);
                    extra.tables[0][0] = lower;
                    // Artificial authenticated numeric payload reaches overflow
                    // handling; this test does not assert it is a valid proof.
                    extra.seal = extra.bits(&family.planes).collect();
                }
            }
            (c.work, c.shared_bytes)
        };
        let mut parts = f.original();
        cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        assert_eq!(parts.problem.rows(), baseline.problem.rows());
        assert_eq!(parts.problem.cols(), baseline.problem.cols());
        let r = cuts.mass_portfolio_report();
        assert_eq!(r.cube.appended_rows, 0);
        assert_eq!(r.phase_appended_rows, 1);
        let c = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
        assert!(c.work >= debt.0);
        assert_eq!(c.shared_bytes, debt.1);
    }
}

#[test]
fn losing_cube_complete_mask_coverage_and_actual_plane_seal_are_required() {
    for k in [2usize, 8] {
        let quote = Quote {
            components: 1,
            negatives: k,
            masks: 1 << k,
            visits: k * (1 << k),
            terms: k,
            work: 0,
            bytes: 0,
        };
        let planes = vec![Plane {
            heads: (0..k).map(|j| (j, -2.)).collect(),
            input: vec![],
            lower: -2.,
        }];
        let component = Component {
            negative: (0..k).collect(),
            positive: vec![],
            positive_units: vec![],
            g: vec![],
            alpha: vec![],
        };
        for count in [0usize, (1 << k) - 1, 1 << k, (1 << k) + 1] {
            let mut c = Collector {
                quote,
                tables: Vec::with_capacity(1),
                activity: vec![None],
                negatives: 0,
                masks: 0,
                visits: 0,
                valid: true,
            };
            c.push(&component, Some(vec![-1.; count]));
            let result = c.finish(&planes);
            assert_eq!(result.is_some(), count == 1 << k);
            if let Some(recipe) = result {
                assert!(recipe.bits(&planes).eq(recipe.seal.iter().copied()));
                let changed = vec![Plane {
                    heads: (0..k).rev().map(|j| (j, -2.)).collect(),
                    input: vec![],
                    lower: -2.,
                }];
                assert!(!recipe.bits(&changed).eq(recipe.seal.iter().copied()));
            }
        }
    }
}

#[test]
fn losing_cube_adjacent_resource_thresholds_preserve_unary_and_final_holds() {
    let p = source(1. / 256.);
    let shared = loser();
    // Permit.context binds the actual cuts address: construct each permit only
    // after the cuts has reached the address at which it will be consumed.
    let quote = {
        let (cuts, winner, mut b, _, signature) = prepared(&p);
        b.phase_all_enabled = true;
        b.phase_cube_enabled = true;
        component_rows::admit(&cuts, &winner, &b).unwrap();
        b.reserve_at_least(shared.cost(&p).unwrap()).unwrap();
        let mut permit = admit(&cuts, &shared, &b, &signature, 1).unwrap();
        b.portfolio_phase_enabled = true;
        conditional::upgrade(&cuts, &shared, &b, 1, &mut permit).unwrap();
        upgrade(&cuts, &shared, &b, 1, &mut permit).unwrap();
        permit.cube.unwrap()
    };
    for (payload, shortfall) in [(false, 1), (false, 0), (true, 1), (true, 0)] {
        let (cuts, winner, mut b, checked, signature) = prepared(&p);
        b.phase_all_enabled = true;
        b.phase_cube_enabled = true;
        component_rows::admit(&cuts, &winner, &b).unwrap();
        b.reserve_at_least(shared.cost(&p).unwrap()).unwrap();
        let mut permit = admit(&cuts, &shared, &b, &signature, 1).unwrap();
        b.portfolio_phase_enabled = true;
        conditional::upgrade(&cuts, &shared, &b, 1, &mut permit).unwrap();
        let before_rows = permit.quote.rows;
        let before = {
            let mut c = b.state.lock().unwrap();
            if payload {
                c.shared_bytes = MAX_BYTES
                    - b.admitted_bytes
                    - c.direction.cache_bytes
                    - c.child.cache_bytes
                    - c.component_rows.admitted_bytes
                    - quote.bytes
                    + shortfall;
            } else {
                c.work = MAX_WORK
                    - c.reserved
                    - c.component_reserved
                    - 2 * permit.quote.cost
                    - 1
                    - quote.work
                    + shortfall;
            }
            (c.work, c.shared_bytes, c.reserved, c.component_reserved)
        };
        let accepted = upgrade(&cuts, &shared, &b, 1, &mut permit).is_some();
        assert_eq!(accepted, shortfall == 0);
        assert!(permit.conditional.is_some());
        assert_eq!(
            permit.quote.rows,
            before_rows + usize::from(accepted) * quote.components
        );
        let after = {
            let c = b.state.lock().unwrap();
            assert_eq!((c.reserved, c.component_reserved), (before.2, before.3));
            assert_eq!(
                c.shared_bytes,
                before.1 + if accepted { quote.bytes } else { 0 }
            );
            assert_eq!(
                c.work,
                before.0
                    + if accepted {
                        quote.work
                    } else {
                        c.portfolio.report.cube.setup_work
                    }
            );
            c.work
        };
        assert!(upgrade(&cuts, &shared, &b, 1, &mut permit).is_none());
        assert_eq!(b.state.lock().unwrap().work, after);
        pay_final(&cuts, &b, permit.quote.cost, Some(&mut permit)).unwrap();
        let checked_loser = check_final(
            &cuts,
            &shared,
            &b,
            Some(permit),
            checked.root_lower,
            &mut || true,
        )
        .unwrap();
        assert!(checked_loser.root_lower < checked.root_lower);
        assert!(report(&b).retained);
    }
}

#[test]
fn losing_cube_single_false_poll_aborts_whole_original_append_without_refund() {
    let f = fixture(79. / 64.);
    let p = f.source();
    let cuts = candidate(&p, true);
    let debt = cuts.mass_portfolio_report().cube.work_units;
    let mut parts = f.original();
    let prefix = parts.problem.clone();
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
    assert_eq!(parts.problem.rows(), prefix.rows());
    assert_eq!(parts.problem.cols(), prefix.cols());
    let r = cuts.mass_portfolio_report();
    assert_eq!(r.cube.work_units, debt);
    assert_eq!(
        (
            r.appended_rows,
            r.phase_appended_rows,
            r.cube.appended_rows,
            r.reserved_ir_rows
        ),
        (0, 0, 0, 0)
    );
}

#[test]
fn losing_cube_every_export_poll_including_both_repairs_is_atomic() {
    let f = fixture(79. / 64.);
    let p = f.source();
    let calibration = candidate(&p, true);
    let mut completed = f.original();
    let mut count = 0;
    calibration
        .append_with_poll(&mut completed, &f.encoding(), end(), &mut || {
            count += usize::from(calibration.mass_portfolio_report().export_attempts == 1);
            true
        })
        .unwrap();
    assert_eq!(calibration.mass_portfolio_report().cube.appended_rows, 1);
    assert_eq!(
        calibration
            .mass_portfolio_report()
            .cube
            .appended_selector_terms,
        2
    );
    assert!(count > 8 && count < 256);
    for target in 1..=count {
        let cuts = candidate(&p, true);
        let mut parts = f.original();
        let original = parts.problem.clone();
        let debt = cuts.mass_portfolio_report().cube.work_units;
        let mut polls = 0;
        let mut stopped = false;
        let result = cuts.append_with_poll(&mut parts, &f.encoding(), end(), &mut || {
            if cuts.mass_portfolio_report().export_attempts == 1 {
                polls += 1;
                if polls == target {
                    stopped = true;
                    return false;
                }
            }
            true
        });
        assert!(stopped && result.is_none(), "export poll {target}/{count}");
        assert_eq!(parts.problem, original, "export poll {target}/{count}");
        let r = cuts.mass_portfolio_report();
        assert_eq!(
            (
                r.appended_rows,
                r.phase_appended_rows,
                r.cube.appended_rows,
                r.reserved_ir_rows
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(r.cube.work_units, debt);
        assert!(
            !cuts
                .signed
                .as_ref()
                .unwrap()
                .budget
                .state
                .lock()
                .unwrap()
                .portfolio
                .pending
        );
    }
}

#[test]
fn losing_cube_header_without_winner_preparation_is_bound_and_once_only() {
    let f = fixture(79. / 64.);
    let p = f.source();
    let cuts = candidate(&p, true);
    let owner = cuts.signed.as_ref().unwrap();
    let mut parts = f.original();
    let original = parts.problem.clone();
    let rows = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
        prepare(owner, &cuts, &parts, binding, end(), &mut || true)
    })
    .unwrap();
    assert_eq!(rows.len(), 4); // two plain, one unary, one full-table row
    assert_eq!(cuts.mass_portfolio_report().cube.prepared_selector_terms, 2);
    assert_eq!(owner.stats().component_rows.phase.prepared_rows, 0);
    assert_eq!(owner.stats().component_rows.phase.cube.prepared_rows, 0);
    let mut foreign = f.original();
    assert!(owner
        .reserve_portfolio_headers(&mut foreign, rows.len())
        .is_none());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, rows.len())
        .is_some());
    assert_eq!(cuts.mass_portfolio_report().reserved_ir_rows, rows.len());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, rows.len())
        .is_none());
    assert_eq!(parts.problem, original);
}
