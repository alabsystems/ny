// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Private malformed-payload/budget controls, not automatic-discovery claims.
use super::super::default_tests::conditional_tests::{candidate, fixture};
use super::super::tests::{loser, prepared, source};
use super::*;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn conditional_portfolio_upgrade_cannot_spend_the_plain_final_pass_hold() {
    for spare in [0, 1] {
        let p = source(1. / 256.);
        let (cuts, _, mut b, checked, signature) = prepared(&p);
        let shared = loser();
        let cost = shared.cost(&p).unwrap();
        b.reserve_at_least(cost).unwrap();
        let mut permit = admit(&cuts, &shared, &b, &signature, 1).unwrap();
        b.portfolio_phase_enabled = true;
        let before = {
            let mut c = b.state.lock().unwrap();
            c.work = MAX_WORK - c.reserved - c.component_reserved - 2 * cost - 1 - spare;
            (c.work, c.shared_bytes, c.reserved, c.component_reserved)
        };
        let bound = permit.quote.rows;
        assert!(upgrade(&cuts, &shared, &b, 1, &mut permit).is_none());
        assert!(permit.conditional.is_none());
        assert_eq!(permit.quote.rows, bound);
        {
            let c = b.state.lock().unwrap();
            assert_eq!(
                (c.work, c.shared_bytes, c.reserved, c.component_reserved),
                before
            );
            assert!(c.portfolio.report.phase_attempted);
            assert!(!c.portfolio.report.phase_admitted);
        }
        pay_final(&cuts, &b, cost, Some(&mut permit)).unwrap();
        let result = check_final(
            &cuts,
            &shared,
            &b,
            Some(permit),
            checked.root_lower,
            &mut || true,
        )
        .unwrap();
        assert!(result.root_lower < checked.root_lower);
        assert!(report(&b).retained);
    }
}

#[test]
fn conditional_portfolio_numeric_or_seal_refusal_preserves_all_plain_rows() {
    for mode in 0..3 {
        let f = fixture();
        let p = f.source();
        let cuts = candidate(&p);
        {
            let mut c = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
            let recipe = c.portfolio.recipe.as_mut().unwrap();
            let extra = recipe.conditional.as_mut().unwrap();
            match mode {
                0 => extra.values[0][0][0] = f64::NAN,
                1 => extra.seal[0] ^= 1,
                _ => {
                    // Synthetic finite minima whose doubled gaps overflow.
                    // Authenticate the artificial payload to reach numeric refusal.
                    for (pairs, plane) in extra.values.iter_mut().zip(&recipe.planes) {
                        pairs.fill([plane.lower * 0.5, f64::MAX]);
                    }
                    extra.seal = extra.bits(&recipe.planes).collect();
                }
            }
        }
        let mut parts = f.original();
        cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        let r = cuts.mass_portfolio_report();
        assert_eq!((r.appended_rows, r.phase_appended_rows), (11, 0));
        let baseline = propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt(
            &p,
            end(),
            Duration::ZERO,
        )
        .cuts
        .unwrap();
        let mut old = f.original();
        baseline.append_to(&mut old, &f.encoding(), end()).unwrap();
        assert_eq!(parts.problem.rows(), old.problem.rows());
        assert_eq!(parts.problem.cols(), old.problem.cols());
    }
}

#[test]
fn conditional_portfolio_seal_binds_the_actual_base_plane() {
    let f = fixture();
    let p = f.source();
    let cuts = candidate(&p);
    let mut c = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
    let recipe = c.portfolio.recipe.as_mut().unwrap();
    assert!(recipe
        .conditional
        .as_ref()
        .unwrap()
        .bits(&recipe.planes)
        .eq(recipe.conditional.as_ref().unwrap().seal.iter().copied()));
    recipe.planes[0].input[0] = 0.25;
    assert!(!recipe
        .conditional
        .as_ref()
        .unwrap()
        .bits(&recipe.planes)
        .eq(recipe.conditional.as_ref().unwrap().seal.iter().copied()));
}

#[test]
fn conditional_portfolio_incomplete_coverage_mints_no_recipe() {
    let f = fixture();
    let p = f.source();
    let cuts = candidate(&p);
    let c = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
    let recipe = c.portfolio.recipe.as_ref().unwrap();
    let quote = recipe.conditional.as_ref().unwrap().quote;
    let mut collector = Collector {
        quote,
        values: Vec::new(),
        negatives: 0,
        visits: 0,
        valid: true,
        cube: None,
    };
    let component = &cuts.signed.as_ref().unwrap().plan.components[0];
    collector.push(
        component,
        Some(vec![[0., 0.]; component.negative.len()]),
        &[[0; 2]; 8],
        None,
    );
    assert!(collector.finish(&recipe.planes).is_none());
}

#[test]
fn conditional_portfolio_foreign_original_context_refuses_without_mutation() {
    let f = fixture();
    let p = f.source();
    let cuts = candidate(&p);
    let mut parts = f.original();
    parts.problem.add_row(-1., 1., [(parts.input_vars[0], 1.)]);
    let before = parts.problem.clone();
    assert!(cuts.append_to(&mut parts, &f.encoding(), end()).is_none());
    assert_eq!(parts.problem.rows(), before.rows());
    assert_eq!(parts.problem.cols(), before.cols());
    assert_eq!(cuts.mass_portfolio_report().phase_appended_rows, 0);
}

#[test]
fn conditional_portfolio_header_fallback_without_any_winner_rows_is_once_only() {
    let f = fixture();
    let p = f.source();
    let cuts = candidate(&p);
    let owner = cuts.signed.as_ref().unwrap();
    let mut parts = f.original();
    let original = parts.problem.clone();
    // Stage only the authentic losing recipe. Neither winner producer is run,
    // so the aggregate phase owner has no pending rows or header grant.
    let rows = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
        prepare(owner, &cuts, &parts, binding, end(), &mut || true)
    })
    .unwrap();
    assert!(rows.len() > 11);
    assert_eq!(owner.stats().component_rows.phase.prepared_rows, 0);
    assert_eq!(owner.stats().component_rows.phase.cube.prepared_rows, 0);
    let mut foreign = f.original();
    assert!(owner
        .reserve_portfolio_headers(&mut foreign, rows.len())
        .is_none());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, rows.len())
        .is_some());
    assert_eq!(parts.problem, original);
    assert_eq!(cuts.mass_portfolio_report().reserved_ir_rows, rows.len());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, rows.len())
        .is_none());
    assert_eq!(parts.problem, original);
}
