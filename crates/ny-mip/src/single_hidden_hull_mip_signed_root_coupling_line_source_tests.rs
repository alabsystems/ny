// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied Plan/Ticket controls through production steering and whole Source.
//! The simple original LP may already solve this graph; no default-discovery claim.
use super::test_fixture::end;
use super::*;
const EPS: f64 = 1. / 128.;
fn source(bias: f64) -> HullMarginProblem {
    let expected = [1_f64, 1., -1., -EPS];
    let mut out = vec![0.; 8];
    for (j, value) in expected.into_iter().enumerate() {
        out[j] = value.next_up().next_up();
    }
    let p = HullMarginProblem::new(
        vec![1., 0., 0., -1., 0., 0., 0., 1., 0., 0., 0., 1.],
        vec![0.5, 0.5, 0., 0.],
        3,
        &out,
        &[bias.next_up().next_up(), 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1., -0.5, -1.],
        vec![1., 0.5, 1.],
    )
    .unwrap();
    assert_eq!(p.c, expected);
    assert_eq!(p.d, bias);
    p
}
fn baseline(
    p: &HullMarginProblem,
) -> (
    HullRootMipCuts<'_>,
    Plan,
    Budget,
    certificate::Checked,
    usize,
) {
    let m = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let cuts = prepare(p, m, lower, end(), &mut || true).unwrap();
    let plan = Plan {
        components: vec![
            Component {
                negative: vec![2],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.; 3],
                alpha: vec![0.; 4],
            },
            Component {
                negative: vec![3],
                positive: vec![],
                positive_units: vec![],
                g: vec![0.; 3],
                alpha: vec![],
            },
        ],
        leftover: vec![],
    };
    plan.valid(&cuts).unwrap();
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    (cuts, plan, b, checked, cost)
}
fn paid_union(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &mut Budget,
    cost: usize,
) -> (Arc<Ticket>, resources::Choice, geometry::Built) {
    let (t, _) = admit(cuts, plan, cost, b).unwrap();
    t.begin(cuts, b).unwrap();
    t.claim_selection(cuts, b, plan, cost).unwrap();
    let check = cost + resources::component_cost(3, 2, 2, 2).unwrap();
    let search = resources::search_cost(3, 2, 2, 2).unwrap();
    let choice = resources::Choice {
        left: 0,
        right: 1,
        selected: 2,
        positive: 2,
        recipient: 0,
        active: 2,
        check,
        search,
        total: check + search,
        score: 1.,
    };
    t.bind_union_choice(cuts, b, &choice, &[choice.left, choice.right])
        .unwrap();
    t.pay(cuts, b, choice.total).unwrap();
    let built = geometry::build(cuts, plan, &choice, b, &t, &mut || true).unwrap();
    assert_eq!((built.selected, built.plan.components.len()), (0, 1));
    assert!(built.recipient.is_none());
    (t, choice, built)
}
#[test]
fn coordinated_line_production_ticket_changes_one_checked_whole_bound_inside_the_original_debit() {
    for bias in [-0.25, -0.75, EPS - 0.5] {
        let p = source(bias);
        let (cuts, plan, mut b, old, cost) = baseline(&p);
        let old_work = b.state.lock().unwrap().work;
        let old_checks = b.state.lock().unwrap().checks;
        let old_lower = old.root_lower;
        let (t, choice, mut built) = paid_union(&cuts, &plan, &mut b, cost);
        let private = t.remaining(&cuts, &b).unwrap();
        let bytes = b.payload_bytes().unwrap();
        let checked_masks = b.state.lock().unwrap().checked_masks;
        let active = steering::optimize(
            &cuts,
            &mut built.plan.components[0],
            2,
            SearchRole::Selected,
            &b,
            &t,
            &mut || true,
        )
        .unwrap();
        assert_eq!(active, 0); // the one joint line replaces both paid sweeps here.
        assert!(built.plan.components[0].alpha.iter().all(|v| *v == 1.));
        let r = b.state.lock().unwrap().root_coupling.clone();
        assert_eq!(
            (
                r.line_mask_attempts,
                r.line_masks_completed,
                r.line_masks_improved,
                r.coordinate_visits
            ),
            (4, 4, 4, 0)
        );
        assert_eq!(t.remaining(&cuts, &b), Some(private));
        assert_eq!(b.payload_bytes(), Some(bytes));
        assert_eq!(b.state.lock().unwrap().work, old_work + LIMIT);
        assert_eq!(b.state.lock().unwrap().checks, old_checks + 1);
        t.claim_checker(&cuts, &b).unwrap();
        let checked = certificate::check(&cuts, &built.plan, &b, t.end, &mut || true).unwrap();
        let exact = bias + 0.5 - EPS;
        assert!(checked.root_lower > exact - 1e-9 && checked.root_lower <= exact);
        assert!(checked.root_lower > old_lower + 0.999999);
        if bias == -0.25 {
            assert!(old_lower < 0. && checked.root_lower > 0.24);
        }
        if bias == EPS - 0.5 {
            assert!(checked.root_lower <= 0.);
        }
        assert!(built.plan.cost(&p).unwrap() <= choice.check);
        assert_eq!(b.state.lock().unwrap().checked_masks - checked_masks, 4);
        assert_eq!(b.state.lock().unwrap().replay_attempts, 0);
        assert!(t.claim_checker(&cuts, &b).is_none());
        // Validity is about the original Source margin, including unsafe/tie
        // points; nominal improvement never mints an opaque publication proof.
        for x in [-1_f64, 0., 1.] {
            for y in [-0.5_f64, 0., 0.5] {
                for z in [-1_f64, 0., 1.] {
                    let actual =
                        bias + (x + 0.5).max(0.) + (-x + 0.5).max(0.) - y.max(0.) - EPS * z.max(0.);
                    assert!(checked.root_lower <= actual);
                }
            }
        }
        let mut polls = 0;
        assert!(steering::optimize(
            &cuts,
            &mut built.plan.components[0],
            2,
            SearchRole::Selected,
            &b,
            &t,
            &mut || {
                polls += 1;
                true
            }
        )
        .is_none());
        assert_eq!(polls, 0);
        assert_eq!(b.state.lock().unwrap().work, old_work + LIMIT);
    }
}
#[test]
fn coordinated_line_production_cancellation_keeps_the_old_owner_and_irreversible_paid_holds() {
    let p = source(-0.25);
    let (cuts, plan, mut b, old, cost) = baseline(&p);
    let witness_before = witness(&plan);
    let lower_before = old.root_lower;
    let (t, _choice, mut built) = paid_union(&cuts, &plan, &mut b, cost);
    let work = b.state.lock().unwrap().work;
    let remaining = t.remaining(&cuts, &b).unwrap();
    let bytes = b.payload_bytes().unwrap();
    let mut saw_line = false;
    assert!(steering::optimize(
        &cuts,
        &mut built.plan.components[0],
        2,
        SearchRole::Selected,
        &b,
        &t,
        &mut || {
            if b.state.lock().unwrap().root_coupling.line_mask_attempts > 0 {
                saw_line = true;
                false
            } else {
                true
            }
        }
    )
    .is_none());
    assert!(saw_line);
    assert_eq!(witness(&plan), witness_before);
    assert_eq!(old.root_lower.to_bits(), lower_before.to_bits());
    assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 0);
    assert_eq!(b.state.lock().unwrap().work, work);
    assert_eq!(b.payload_bytes(), Some(bytes));
    assert_eq!(t.remaining(&cuts, &b), Some(remaining));
    assert!(b.state.lock().unwrap().reserved >= 2 * cost);
    let mut polls = 0;
    assert!(steering::optimize(
        &cuts,
        &mut built.plan.components[0],
        2,
        SearchRole::Selected,
        &b,
        &t,
        &mut || {
            polls += 1;
            true
        }
    )
    .is_none());
    assert_eq!(polls, 0);
    assert_eq!(b.state.lock().unwrap().replay_attempts, 0);
}
