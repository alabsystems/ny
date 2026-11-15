// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied checked-owner controls, not default partition/LP-discovery evidence.
use super::*;

const EPS: f64 = 1. / 256.;
const DELTA: f64 = 1. / 8.;
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn source(bias: f64) -> HullMarginProblem {
    let output = [
        1_f64.next_up().next_up(),
        1_f64.next_up().next_up(),
        (-1_f64).next_up().next_up(),
        (-1_f64).next_up().next_up(),
        0.,
        0.,
        0.,
        0.,
    ];
    let p = HullMarginProblem::new(
        vec![1., 0., 0., -1., EPS, 0., 1., 0., EPS, -1., 2. * EPS, 0.],
        vec![0.; 4],
        3,
        &output,
        &[bias.next_up().next_up(), 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; 3],
        vec![1.; 3],
    )
    .unwrap();
    assert_eq!(p.d, bias);
    assert_eq!(p.c, [1., 1., -1., -1.]);
    p
}
pub(super) fn winner() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: vec![2],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.; 3],
                alpha: vec![0., 1.],
            },
            Component {
                negative: vec![3],
                positive: vec![1],
                positive_units: vec![2],
                g: vec![0.; 3],
                alpha: vec![0., 1.],
            },
        ],
        leftover: vec![],
    }
}
pub(super) fn loser() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: vec![2],
                positive: vec![0, 1],
                positive_units: vec![1, 1],
                g: vec![-0.5 + DELTA, 0., 0.],
                alpha: vec![0., 1., 1., 0.],
            },
            Component {
                negative: vec![3],
                positive: vec![0, 1],
                positive_units: vec![1, 1],
                g: vec![0.5, 0., 0.],
                alpha: vec![1., 0., 0., 1.],
            },
        ],
        leftover: vec![],
    }
}
#[allow(
    clippy::rc_buffer,
    reason = "Keep the production signature allocation and ownership layout in this fixture."
)]
pub(super) fn prepared(
    p: &HullMarginProblem,
) -> (
    HullRootMipCuts<'_>,
    Plan,
    Budget,
    certificate::Checked,
    Arc<RetainedWords>,
) {
    let multipliers = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &multipliers).unwrap();
    let cuts = super::super::super::prepare(p, multipliers, lower, end(), &mut || true).unwrap();
    let plan = winner();
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    // Actual existing root/sharing ledgers, with a supplied Plan. No proposal
    // or default discovery credit is inferred from this private owner setup.
    root_coupling::admit(&cuts, &plan, cost, &mut b).unwrap();
    b.admit_sharing().unwrap();
    b.portfolio_enabled = true;
    let signature = Arc::new(signature(&cuts).collect());
    (cuts, plan, b, checked, signature)
}

#[test]
fn portfolio_distinct_affine_losing_scalar_adds_a_strict_valid_component_plane() {
    for bias in [3. * EPS, EPS, 2. * EPS] {
        let p = source(bias);
        let (cuts, old, b, checked, signature) = prepared(&p);
        let old_bits = witness(&old);
        let old_lower = checked.root_lower;
        let shared = loser();
        let cost = shared.cost(&p).unwrap();
        b.reserve_at_least(cost).unwrap();
        let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
        let before = b.state.lock().unwrap().checks;
        pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
        let result = check_final(&cuts, &shared, &b, permit, old_lower, &mut || true).unwrap();
        assert_eq!(b.state.lock().unwrap().checks, before + 1);
        assert!(result.root_lower < old_lower);
        assert!((result.root_lower - (bias - 2. * DELTA - 2.5 * EPS)).abs() < 1e-9);
        assert_eq!(witness(&old), old_bits);
        assert_eq!(checked.root_lower.to_bits(), old_lower.to_bits());
        let c = b.state.lock().unwrap();
        let recipe = c.portfolio.recipe.as_ref().unwrap();
        assert!(c.portfolio.report.retained);
        assert_eq!(c.portfolio.report.completed_collections, 1);
        assert_eq!(recipe.planes.len(), 2);
        // All four affine preactivations are zero at the original input.
        // Independent triangles admit P0=N0=.5,P1=N1=0 (selectors .5,0,.5,0).
        let fake = [0.5, 0., 0.5, 0.];
        let plane = &recipe.planes[0];
        let lhs = plane.heads.iter().map(|(j, c)| c * fake[*j]).sum::<f64>();
        assert!(lhs < plane.lower);
        // Winner component and phase rows allow zero head differences.
        assert_eq!(fake[0] - fake[2], 0.);
        assert_eq!(fake[1] - fake[3], 0.);
        for x in [-1_f64, 0., 1.] {
            for y in [-1_f64, 0., 1.] {
                for z in [-1_f64, 0., 1.] {
                    let input = [x, y, z];
                    let heads = [
                        x.max(0.),
                        (-x + EPS * y).max(0.),
                        (x + EPS * z).max(0.),
                        (-x + 2. * EPS * y).max(0.),
                    ];
                    for plane in &recipe.planes {
                        let value = plane.heads.iter().map(|(j, c)| c * heads[*j]).sum::<f64>()
                            + plane
                                .input
                                .iter()
                                .zip(input)
                                .map(|(c, x)| c * x)
                                .sum::<f64>();
                        assert!(value >= plane.lower);
                    }
                }
            }
        }
        // Two Lipschitz gaps attain the exact graph minimum d-2epsilon.
        let attaining = bias + 0. + EPS - EPS - 2. * EPS;
        assert_eq!(attaining, bias - 2. * EPS);
        assert_eq!(attaining > 0., bias == 3. * EPS);
        assert_eq!(attaining == 0., bias == 2. * EPS);
    }
}

#[test]
fn portfolio_one_paid_final_check_and_duplicate_admission_cannot_renew_credit() {
    let p = source(EPS);
    let (cuts, _, b, checked, signature) = prepared(&p);
    let shared = loser();
    let cost = shared.cost(&p).unwrap();
    b.reserve_at_least(cost).unwrap();
    let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
    let work = b.state.lock().unwrap().work;
    assert!(admit(&cuts, &shared, &b, &signature, 1).is_none());
    assert_eq!(b.state.lock().unwrap().work, work);
    pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
    let paid = b.state.lock().unwrap().checks;
    assert!(pay_final(&cuts, &b, cost, permit.as_mut()).is_none());
    assert_eq!(b.state.lock().unwrap().checks, paid);
    // Already paid final authority survives exhausting the remaining slots;
    // collection cannot allocate an additional checker or publication pass.
    b.state.lock().unwrap().checks = MAX_CHECKS;
    check_final(&cuts, &shared, &b, permit, checked.root_lower, &mut || true).unwrap();
    assert_eq!(b.state.lock().unwrap().checks, MAX_CHECKS);
    assert_eq!(report(&b).collection_attempts, 1);
    assert_eq!(b.state.lock().unwrap().source_replays, 0);
}

#[test]
fn portfolio_unpaid_foreign_and_cancelled_final_pass_keep_older_owner() {
    for kind in 0..3 {
        let p = source(EPS);
        let (cuts, old, b, checked, signature) = prepared(&p);
        let (foreign, _, _, _, _) = prepared(&p);
        let shared = loser();
        b.reserve_at_least(shared.cost(&p).unwrap()).unwrap();
        let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
        let debt = b.state.lock().unwrap().work;
        let old_bits = witness(&old);
        let lower = checked.root_lower.to_bits();
        if kind != 0 {
            pay_final(&cuts, &b, shared.cost(&p).unwrap(), permit.as_mut()).unwrap();
        }
        let result = check_final(
            if kind == 1 { &foreign } else { &cuts },
            &shared,
            &b,
            permit,
            checked.root_lower,
            &mut || kind != 2,
        );
        assert!(result.is_none());
        assert_eq!(witness(&old), old_bits);
        assert_eq!(checked.root_lower.to_bits(), lower);
        assert!(b.state.lock().unwrap().work >= debt);
        assert!(!report(&b).retained);
        assert_eq!(report(&b).appended_rows, 0);
    }
}

#[test]
fn portfolio_selected_or_equal_scalar_does_not_duplicate_normal_owner_rows() {
    let p = source(EPS);
    let (cuts, _, b, _, signature) = prepared(&p);
    let shared = loser();
    let cost = shared.cost(&p).unwrap();
    b.reserve_at_least(cost).unwrap();
    let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
    pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
    let checked = check_final(&cuts, &shared, &b, permit, -1., &mut || true).unwrap();
    assert!(checked.root_lower > -1.);
    let r = report(&b);
    assert_eq!(r.completed_collections, 1);
    assert!(!r.retained);
    assert_eq!(r.stop_reason, "not-a-strict-scalar-loser");
    assert!(b.state.lock().unwrap().portfolio.recipe.is_none());
}

#[test]
fn portfolio_shape_quote_adds_persistent_bytes_once_and_protects_existing_holds() {
    let p = source(EPS);
    let (cuts, _, b, _, signature) = prepared(&p);
    let shared = loser();
    let q = quote(&cuts, &shared, &b).unwrap();
    let (work, bytes, reserved, optional) = {
        let c = b.state.lock().unwrap();
        (c.work, c.shared_bytes, c.reserved, c.component_reserved)
    };
    let original_total = b.payload_bytes().unwrap();
    let permit = admit(&cuts, &shared, &b, &signature, 1).unwrap();
    let c = b.state.lock().unwrap();
    assert_eq!(c.work, work + q.work);
    assert_eq!(c.shared_bytes, bytes + q.bytes);
    assert_eq!((c.reserved, c.component_reserved), (reserved, optional));
    assert!(!permit.final_paid);
    drop(c);
    assert_eq!(b.payload_bytes().unwrap(), original_total + q.bytes);
    assert!(b.payload_bytes().unwrap() <= MAX_BYTES);
    assert!(quote(
        &cuts,
        &shared,
        &Budget {
            dimensions: (usize::MAX, 4, 2),
            ..b.clone()
        }
    )
    .is_none());
}

#[test]
fn portfolio_partial_or_final_mask_cancellation_retains_debt_and_no_recipe() {
    for completed in [3usize, 4] {
        let p = source(EPS);
        let (cuts, old, b, checked, signature) = prepared(&p);
        let shared = loser();
        let cost = shared.cost(&p).unwrap();
        b.reserve_at_least(cost).unwrap();
        let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
        pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
        let (masks, debt, checks, holds) = {
            let c = b.state.lock().unwrap();
            (
                c.checked_masks,
                c.work,
                c.checks,
                (c.reserved, c.component_reserved),
            )
        };
        let bits = witness(&old);
        let lower = checked.root_lower.to_bits();
        let result = check_final(&cuts, &shared, &b, permit, checked.root_lower, &mut || {
            b.state.lock().unwrap().checked_masks < masks + completed
        });
        assert!(result.is_none());
        let c = b.state.lock().unwrap();
        assert_eq!(c.checked_masks, masks + completed);
        assert_eq!(
            (c.work, c.checks, c.reserved, c.component_reserved),
            (debt, checks, holds.0, holds.1)
        );
        assert!(c.portfolio.recipe.is_none());
        assert_eq!(c.portfolio.report.completed_collections, 0);
        assert_eq!(witness(&old), bits);
        assert_eq!(checked.root_lower.to_bits(), lower);
    }
}

#[test]
fn portfolio_adjacent_work_and_payload_quota_refusals_preserve_older_holds() {
    for payload in [false, true] {
        for gap in [0usize, 1] {
            let p = source(EPS);
            let (cuts, _, b, _, signature) = prepared(&p);
            let shared = loser();
            let cost = shared.cost(&p).unwrap();
            b.reserve_at_least(cost).unwrap();
            let q = quote(&cuts, &shared, &b).unwrap();
            let holds;
            {
                let mut c = b.state.lock().unwrap();
                holds = (c.reserved, c.component_reserved);
                // Synthetic occupancy changes the ledger only; no allocation or
                // real search-completion claim is attached to these boundaries.
                if payload {
                    let base = b.admitted_bytes
                        + c.shared_bytes
                        + c.direction.cache_bytes
                        + c.component_rows.admitted_bytes
                        + q.bytes;
                    c.child.cache_bytes = MAX_BYTES - base + gap;
                } else {
                    c.work =
                        MAX_WORK - c.reserved - c.component_reserved - q.work - 2 * cost - 1 + gap;
                }
            }
            let permit = admit(&cuts, &shared, &b, &signature, 1);
            assert_eq!(permit.is_some(), gap == 0);
            let c = b.state.lock().unwrap();
            assert_eq!((c.reserved, c.component_reserved), holds);
            assert_eq!(c.portfolio.report.admitted, gap == 0);
            assert_eq!(
                c.portfolio.report.admitted_bytes,
                if gap == 0 { q.bytes } else { 0 }
            );
            assert_eq!(
                c.portfolio.report.work_units,
                if gap == 0 {
                    q.work
                } else {
                    (shared.components.len() + b.dimensions.1 + b.dimensions.0 + 1) * 64
                }
            );
            assert!(c.portfolio.recipe.is_none());
        }
    }
}

#[test]
fn portfolio_finite_source_but_doubled_row_overflow_keeps_checked_scalar() {
    let mut p = source(EPS);
    // Synthetic finite stable heads; no LP/native-solver completion claim.
    let big = f64::MAX * 0.75;
    p.w1.fill(0.);
    p.b1.copy_from_slice(&[big, 0., big, 0.]);
    let (cuts, _, b, old, signature) = prepared(&p);
    let mut shared = loser();
    shared.components[0].alpha.fill(0.);
    shared.components[1].alpha = vec![1., 0., 1., 0.];
    let cost = shared.cost(&p).unwrap();
    b.reserve_at_least(cost).unwrap();
    let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
    pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
    let before = b.state.lock().unwrap().checked_masks;
    let checked = check_final(&cuts, &shared, &b, permit, old.root_lower, &mut || true).unwrap();
    assert_eq!(b.state.lock().unwrap().checked_masks, before + 4);
    let minima = checked.component_minima.as_ref().unwrap();
    assert!(minima.iter().all(|value| value.is_finite()));
    assert!(minima[0] < -f64::MAX / 2. && minima[1] > 0.);
    assert!(checked.root_lower.is_finite() && checked.root_lower < old.root_lower);
    assert_eq!(
        checked.component_minima.as_ref().unwrap()[0] * 2.,
        f64::NEG_INFINITY
    );
    let r = report(&b);
    assert_eq!(r.completed_collections, 1);
    assert!(!r.retained);
    assert_eq!(r.stop_reason, "recipe-refused");
    assert_eq!(r.appended_rows, 0);
    assert!(b.state.lock().unwrap().portfolio.recipe.is_none());
}

#[test]
fn portfolio_original_ir_header_envelope_reserves_once_without_changing_model() {
    assert!(size_of::<RowSpec>() <= 40);
    for spare in [0, 4096] {
        let f = default_tests::Fixture::supplied(EPS);
        let p = f.source();
        let cuts = default_tests::supplied_cuts(&p);
        let owner = cuts.signed.as_ref().unwrap();
        let mut parts = f.original();
        // The synthetic spare capacity predates the new publication. A large
        // existing buffer needs no move; a small one is covered by I+R.
        parts.problem.reserve_rows_exact(spare);
        let original = parts.problem.clone();
        let encoding = f.encoding();
        let rows = binding::with_phase_binding_test(&cuts, &parts, &encoding, |binding| {
            prepare(owner, &cuts, &parts, binding, end(), &mut || true)
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        let (limit, bound) = {
            let c = owner.budget.state.lock().unwrap();
            (
                c.portfolio.report.ir_row_limit,
                c.portfolio.prepared_row_bound,
            )
        };
        assert_eq!(
            limit,
            8 * (owner.budget.dimensions.0 + owner.budget.dimensions.1 + 1)
        );
        let mut foreign = f.original();
        assert!(reserve_ir_headers(owner, &mut foreign, rows.len()).is_none());
        assert!(reserve_ir_headers(owner, &mut parts, bound + 1).is_none());
        assert_eq!(report(&owner.budget).reserved_ir_rows, 0);
        assert!(reserve_ir_headers(owner, &mut parts, rows.len()).is_some());
        assert_eq!(parts.problem, original); // rows, columns and margin marker
        assert_eq!(report(&owner.budget).reserved_ir_rows, rows.len());
        assert!(reserve_ir_headers(owner, &mut parts, rows.len()).is_none());
        assert_eq!(parts.problem, original);
    }
    // Many unused ORIGINAL output rows enlarge the authentic encoder while
    // preserving the selected two-output Source margin. No forged token or
    // mutation of authenticated Parts is used for the I/I+1 boundary.
    for over in [0, 1] {
        let f = default_tests::Fixture::supplied(EPS);
        let p = f.source();
        let cuts = default_tests::supplied_cuts(&p);
        let owner = cuts.signed.as_ref().unwrap();
        let limit = report(&owner.budget).ir_row_limit;
        let n = 2 + limit + over - f.original().problem.num_rows();
        let f = f.with_unused_outputs(n);
        let parts = f.original();
        assert_eq!(parts.problem.num_rows(), limit + over);
        let original = parts.problem.clone();
        let encoding = f.encoding();
        let rows = binding::with_phase_binding_test(&cuts, &parts, &encoding, |binding| {
            prepare(owner, &cuts, &parts, binding, end(), &mut || true)
        })
        .expect("complete original model binding succeeds for both sizes");
        assert_eq!(rows.len(), if over == 0 { 2 } else { 0 });
        assert_eq!(report(&owner.budget).prepared_rows, rows.len());
        assert_eq!(parts.problem, original);
        assert_eq!(report(&owner.budget).export_attempts, 1);
    }
}
