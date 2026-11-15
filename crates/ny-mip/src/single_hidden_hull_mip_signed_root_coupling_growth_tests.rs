// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied-plan controls for bounded union discovery and prepaid capabilities.
use super::test_fixture::end;
use super::*;

fn source(positive: &[f64], negative_weight: f64) -> HullMarginProblem {
    let expected: Vec<_> = positive
        .iter()
        .copied()
        .chain((0..6).map(|_| negative_weight))
        .collect();
    let h = expected.len();
    let mut output = vec![0.; 2 * h];
    for (j, c) in expected.iter().enumerate() {
        output[j] = c.next_up().next_up();
    }
    let p = HullMarginProblem::new(
        vec![1.; h],
        vec![0.; h],
        1,
        &output,
        &[-0.25, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    assert_eq!(p.c, expected);
    p
}

fn singles(p: &HullMarginProblem, allocations: Vec<Vec<(usize, u8)>>) -> Plan {
    let positives = p.c.iter().filter(|c| **c > 0.).count();
    assert_eq!(allocations.len(), 6);
    Plan {
        components: allocations
            .into_iter()
            .enumerate()
            .map(|(i, slots)| Component {
                negative: vec![positives + i],
                positive: slots.iter().map(|(j, _)| *j).collect(),
                positive_units: slots.iter().map(|(_, units)| *units).collect(),
                alpha: vec![0.5; 2 * slots.len()],
                g: vec![0.],
            })
            .collect(),
        leftover: vec![],
    }
}

fn context(p: &HullMarginProblem, plan: Plan) -> (HullRootMipCuts<'_>, Plan, Budget, usize) {
    let mut m = HullMultipliers::zeros(p);
    for (j, c) in p.c.iter().enumerate() {
        if *c > 0. {
            m.alpha[j] = 0.5;
        }
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let cuts = prepare(p, m, lower, end(), &mut || true).unwrap();
    plan.valid(&cuts).unwrap();
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    (cuts, plan, b, cost)
}

fn whole_allocations() -> Vec<Vec<(usize, u8)>> {
    let mut allocation = vec![vec![]; 6];
    allocation[0] = vec![(0, 2)];
    allocation
}

fn generous_choice(cost: usize, selected: usize, positive: usize) -> resources::Choice {
    let check = cost + resources::component_cost(1, selected, positive, positive).unwrap() + 10000;
    let active = positive.min(6);
    let search = resources::search_cost(1, selected, positive, active).unwrap();
    resources::Choice {
        left: 0,
        right: 1,
        selected,
        positive,
        recipient: 0,
        active,
        check,
        search,
        total: check + search,
        score: 1.,
    }
}

#[test]
fn root_coupling_six_parent_discovery_uses_one_paid_geometry_search_and_whole_check() {
    let p = source(&[6.], -1.);
    let plan = singles(&p, whole_allocations());
    let (cuts, plan, mut b, cost) = context(&p, plan);
    let work = b.state.lock().unwrap().work;
    let checks = b.state.lock().unwrap().checks;
    let masks = b.state.lock().unwrap().checked_masks;
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    ticket.begin(&cuts, &b).unwrap();
    let remaining = ticket.remaining(&cuts, &b).unwrap();
    let feature_heads: usize = plan
        .components
        .iter()
        .map(|c| c.positive.len() + c.negative.len())
        .sum();
    let mut polls = 0;
    let choice = resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || {
        polls += 1;
        true
    })
    .unwrap();
    let (parents, count) = ticket.parents(&cuts, &b, &choice).unwrap();
    assert_eq!(count, 6);
    assert_eq!(&parents[..count], &[0, 1, 2, 3, 4, 5]);
    assert_eq!(choice.selected, 6);
    assert!(polls >= feature_heads && polls - feature_heads <= plan.components.len().pow(2));
    assert_eq!(ticket.remaining(&cuts, &b), Some(remaining));
    assert_eq!(b.state.lock().unwrap().checked_masks, masks);
    assert_eq!(b.state.lock().unwrap().checks, checks + 1);
    assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 0);
    assert_eq!(b.state.lock().unwrap().coordinate_proposals, 0);
    ticket.pay(&cuts, &b, choice.total).unwrap();
    let mut built = geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).unwrap();
    assert_eq!(built.plan.components.len(), 1);
    assert_eq!(built.plan.masks(), Some(64));
    assert!(built.recipient.is_none());
    steering::optimize(
        &cuts,
        &mut built.plan.components[built.selected],
        choice.active,
        SearchRole::Selected,
        &b,
        &ticket,
        &mut || true,
    )
    .unwrap();
    ticket.claim_checker(&cuts, &b).unwrap();
    let checked = certificate::check(&cuts, &built.plan, &b, ticket.end, &mut || true).unwrap();
    assert!(checked.root_lower > -0.251 && checked.root_lower <= -0.25);
    assert!(built.plan.cost(&p).unwrap() <= choice.check);
    assert_eq!(b.state.lock().unwrap().checked_masks - masks, 64);
    assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 1);
    assert_eq!(b.state.lock().unwrap().work, work + LIMIT);
    assert!(ticket.claim_checker(&cuts, &b).is_none());
}

#[test]
fn root_coupling_unfundable_growth_retains_the_funded_seed_without_an_extra_check() {
    let p = source(&[6.], -1.);
    let (cuts, plan, mut b, cost) = context(&p, singles(&p, whole_allocations()));
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    ticket.begin(&cuts, &b).unwrap();
    let removed = resources::component_cost(1, 1, 1, 0).unwrap()
        + resources::component_cost(1, 1, 0, 0).unwrap();
    let pair_check = cost - removed + resources::component_cost(1, 2, 1, 1).unwrap();
    let pair_total = pair_check + resources::search_cost(1, 2, 1, 1).unwrap();
    ticket
        .pay(&cuts, &b, ticket.remaining(&cuts, &b).unwrap() - pair_total)
        .unwrap();
    let masks = b.state.lock().unwrap().checked_masks;
    let choice = resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).unwrap();
    let (_, count) = ticket.parents(&cuts, &b, &choice).unwrap();
    assert_eq!((count, choice.selected), (2, 2));
    assert_eq!(choice.total, pair_total);
    assert_eq!(b.state.lock().unwrap().checked_masks, masks);
    assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 0);
    ticket.pay(&cuts, &b, choice.total).unwrap();
    assert_eq!(ticket.remaining(&cuts, &b), Some(0));
    let built = geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).unwrap();
    built.plan.valid(&cuts).unwrap();
    assert_eq!(built.plan.components[built.selected].negative.len(), 2);
}

#[test]
fn root_coupling_union_parent_membership_choice_and_private_seal_reject_rebinding() {
    let p = source(&[6.], -1.);
    let (cuts, plan, mut b, cost) = context(&p, singles(&p, whole_allocations()));
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    ticket.begin(&cuts, &b).unwrap();
    ticket.claim_selection(&cuts, &b, &plan, cost).unwrap();
    let choice = generous_choice(cost, 6, 1);
    for invalid in [
        vec![],
        vec![0],
        vec![0, 0],
        vec![1, 0],
        vec![0, 6],
        vec![0, 1, 2, 3, 4, 5, 6],
    ] {
        assert!(ticket
            .bind_union_choice(&cuts, &b, &choice, &invalid)
            .is_none());
    }
    ticket
        .bind_union_choice(&cuts, &b, &choice, &[0, 1, 2, 3, 4, 5])
        .unwrap();
    assert!(ticket
        .bind_union_choice(&cuts, &b, &choice, &[0, 1])
        .is_none());
    let mut changed = choice.clone();
    changed.active = 0;
    assert!(ticket.parents(&cuts, &b, &changed).is_none());
    assert_eq!(ticket.parents(&cuts, &b, &choice).unwrap().1, 6);
    ticket.pay(&cuts, &b, choice.total).unwrap();
    // A still sorted/in-range truncated set cannot replace the frozen six IDs.
    ticket.state.lock().unwrap().parent_count = 5;
    assert!(ticket.parents(&cuts, &b, &choice).is_none());
    let mut polls = 0;
    assert!(
        geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || {
            polls += 1;
            true
        })
        .is_none()
    );
    assert_eq!(polls, 0);
    assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 0);
}

#[test]
fn root_coupling_ninety_six_whole_positives_keep_stable_top_sixteen_and_exact_leftover_mass() {
    let weights: Vec<_> = (1..=96).map(f64::from).collect();
    let p = source(&weights, -10000.);
    let allocations = (0..6)
        .map(|parent| (16 * parent..16 * parent + 16).map(|j| (j, 2)).collect())
        .collect();
    let (cuts, plan, mut b, cost) = context(&p, singles(&p, allocations));
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    assert!(b.payload_bytes().unwrap() <= MAX_BYTES);
    ticket.begin(&cuts, &b).unwrap();
    ticket.claim_selection(&cuts, &b, &plan, cost).unwrap();
    let choice = generous_choice(cost, 6, 16);
    ticket
        .bind_union_choice(&cuts, &b, &choice, &[0, 1, 2, 3, 4, 5])
        .unwrap();
    ticket.pay(&cuts, &b, choice.total).unwrap();
    let built = geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).unwrap();
    built.plan.valid(&cuts).unwrap();
    let selected = &built.plan.components[built.selected];
    assert_eq!(selected.positive, (80..96).rev().collect::<Vec<_>>());
    assert_eq!(built.plan.leftover, (0..80).collect::<Vec<_>>());
    assert!(built.recipient.is_none());
    assert_eq!(selected.g, vec![-1620.]);
    for j in 0..96 {
        let units: usize = built
            .plan
            .components
            .iter()
            .map(|c| {
                c.positive
                    .iter()
                    .position(|k| *k == j)
                    .map_or(0, |slot| usize::from(c.positive_units[slot]))
            })
            .sum::<usize>()
            + 2 * usize::from(built.plan.leftover.contains(&j));
        assert_eq!(units, 2);
    }
    assert!(built.plan.cost(&p).unwrap() <= choice.check);
    ticket.claim_checker(&cuts, &b).unwrap();
    let checked = certificate::check(&cuts, &built.plan, &b, ticket.end, &mut || true).unwrap();
    for x in [-1_f64, 0., 1.] {
        let actual = p.d + p.c.iter().map(|c| c * x.max(0.)).sum::<f64>();
        assert!(checked.root_lower <= actual);
    }
}

#[test]
fn root_coupling_half_unit_seed_cannot_grow_and_preserves_the_other_half_owner() {
    let p = source(&[6.], -1.);
    let mut allocation = vec![vec![]; 6];
    allocation[0] = vec![(0, 1)];
    allocation[1] = vec![(0, 1)];
    let (cuts, plan, mut b, cost) = context(&p, singles(&p, allocation));
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    ticket.begin(&cuts, &b).unwrap();
    let choice = resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).unwrap();
    let (parents, count) = ticket.parents(&cuts, &b, &choice).unwrap();
    assert_eq!((count, choice.selected), (2, 2));
    assert_eq!(parents[..count].iter().filter(|i| **i < 2).count(), 1);
    ticket.pay(&cuts, &b, choice.total).unwrap();
    let built = geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).unwrap();
    built.plan.valid(&cuts).unwrap();
    assert_eq!(built.plan.leftover, [] as [usize; 0]);
    assert_eq!(
        built.plan.components[built.selected].positive_units,
        vec![1]
    );
    let appearances: Vec<_> = built
        .plan
        .components
        .iter()
        .filter_map(|c| {
            c.positive
                .iter()
                .position(|j| *j == 0)
                .map(|slot| c.positive_units[slot])
        })
        .collect();
    assert_eq!(appearances, vec![1, 1]);
    assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 0);
}
