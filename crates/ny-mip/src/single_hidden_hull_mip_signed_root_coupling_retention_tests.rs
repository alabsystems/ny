// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Engineering controls for prepaid debt, ownership and checked-owner retention.
use super::test_fixture::{baseline, end, source};
use super::*;

fn whole() -> Plan {
    Plan {
        components: vec![Component {
            negative: vec![1, 2],
            positive: vec![0],
            positive_units: vec![2],
            g: vec![0.],
            alpha: vec![0., 0.25, 0.75, 1.],
        }],
        leftover: vec![],
    }
}

#[test]
fn root_coupling_debt_payload_and_private_work_never_refund_or_double_charge() {
    let p = source(-0.25);
    let (cuts, plan, mut b, _, cost) = baseline(&p);
    let (meta, payload, _) = resources::admission(&plan, &b).unwrap();
    let before = b.state.lock().unwrap().work;
    let checks = b.state.lock().unwrap().checks;
    let bytes = b.admitted_bytes;
    let authority = (b.authority, b.search_end);
    let (ticket, paid_meta) = admit(&cuts, &plan, cost, &mut b).unwrap();
    assert_eq!(paid_meta, meta);
    assert_eq!(b.state.lock().unwrap().work, before + LIMIT);
    assert_eq!(b.state.lock().unwrap().checks, checks + 1);
    assert_eq!(b.admitted_bytes, bytes + payload);
    assert_eq!((b.authority, b.search_end), authority);
    assert_eq!(ticket.remaining(&cuts, &b), Some(LIMIT - PREFLIGHT - meta));
    assert!(ticket.pay(&cuts, &b, 1).is_none());
    ticket.begin(&cuts, &b).unwrap();
    let choice = resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).unwrap();
    let free = ticket.remaining(&cuts, &b).unwrap();
    ticket.pay(&cuts, &b, choice.total).unwrap();
    assert_eq!(ticket.remaining(&cuts, &b), Some(free - choice.total));
    assert!(ticket.pay(&cuts, &b, LIMIT).is_none());
    assert_eq!(ticket.remaining(&cuts, &b), Some(free - choice.total));
    assert!(admit(&cuts, &plan, cost, &mut b).is_none());
    let c = b.state.lock().unwrap();
    assert_eq!(c.work, before + LIMIT);
    assert_eq!(
        c.root_coupling.used_work_units,
        PREFLIGHT + meta + choice.total
    );
    assert_eq!(c.root_coupling.admitted_bytes, payload);
    assert_eq!(b.admitted_bytes, bytes + payload);
}

#[test]
fn root_coupling_private_stages_and_checker_are_once_only_across_report_reset() {
    let p = source(-0.25);
    let (cuts, plan, mut b, _, cost) = baseline(&p);
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    assert!(ticket.claim_checker(&cuts, &b).is_none());
    ticket.begin(&cuts, &b).unwrap();
    assert!(ticket.begin(&cuts, &b).is_none());
    let choice = resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).unwrap();
    let mut polls = 0;
    assert!(resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert!(ticket.claim_checker(&cuts, &b).is_none());
    assert!(
        geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || {
            polls += 1;
            true
        })
        .is_none()
    );
    assert_eq!(polls, 0);
    ticket.pay(&cuts, &b, choice.total).unwrap();
    let mut changed_choice = choice.clone();
    changed_choice.active += 1;
    assert!(
        geometry::build(&cuts, &plan, &changed_choice, &b, &ticket, &mut || {
            polls += 1;
            true
        })
        .is_none()
    );
    assert_eq!(polls, 0);
    let mut built = geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).unwrap();
    assert!(built.plan.valid(&cuts).is_some());
    assert!(
        geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || {
            polls += 1;
            true
        })
        .is_none()
    );
    assert_eq!(polls, 0);
    let selected = &mut built.plan.components[built.selected];
    steering::optimize(
        &cuts,
        selected,
        choice.active,
        SearchRole::Selected,
        &b,
        &ticket,
        &mut || true,
    )
    .unwrap();
    assert!(steering::optimize(
        &cuts,
        selected,
        choice.active,
        SearchRole::Selected,
        &b,
        &ticket,
        &mut || {
            polls += 1;
            true
        }
    )
    .is_none());
    assert_eq!(polls, 0);
    ticket.claim_checker(&cuts, &b).unwrap();
    let work = b.state.lock().unwrap().work;
    b.state.lock().unwrap().root_coupling = SignedRootCouplingReport::default();
    assert!(ticket.begin(&cuts, &b).is_none());
    assert!(ticket.claim_checker(&cuts, &b).is_none());
    assert!(resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).is_none());
    assert!(geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).is_none());
    assert!(steering::optimize(
        &cuts,
        selected,
        choice.active,
        SearchRole::Selected,
        &b,
        &ticket,
        &mut || true
    )
    .is_none());
    assert_eq!(b.state.lock().unwrap().work, work);
    assert!(admit(&cuts, &plan, cost, &mut b).is_none());
}

#[test]
fn root_coupling_ticket_binds_source_context_ledger_clock_payload_and_cutoff() {
    let p = source(-0.25);
    let other = source(-0.25);
    let (cuts, plan, mut b, _, cost) = baseline(&p);
    let (foreign, _, _, _, _) = baseline(&other);
    let (same_source, _, _, _, _) = baseline(&p);
    let (mut ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    assert!(ticket.owned(&cuts, &b));
    assert!(!ticket.owned(&foreign, &b));
    assert!(!ticket.owned(&same_source, &b));
    ticket.begin(&cuts, &b).unwrap();
    let mut foreign_polls = 0;
    assert!(
        resources::choose(&same_source, &plan, &b, &ticket, cost, &mut || {
            foreign_polls += 1;
            true
        })
        .is_none()
    );
    assert_eq!(foreign_polls, 0);
    let original = b.clone();
    b.state = Arc::new(Mutex::new(Counts::default()));
    assert!(!ticket.owned(&cuts, &b));
    b = original.clone();
    b.source ^= 1;
    assert!(!ticket.owned(&cuts, &b));
    b = original.clone();
    b.dimensions.2 += 1;
    assert!(!ticket.owned(&cuts, &b));
    b = original.clone();
    b.authority -= Duration::from_millis(1);
    assert!(!ticket.owned(&cuts, &b));
    b = original.clone();
    b.search_end += Duration::from_millis(1);
    assert!(!ticket.owned(&cuts, &b));
    b = original.clone();
    b.admitted_bytes -= 1;
    assert!(!ticket.owned(&cuts, &b));
    b = original;
    assert!(ticket.owned(&cuts, &b));
    b.root_coupling = None;
    let inner = Arc::get_mut(&mut ticket).unwrap();
    inner.end += Duration::from_millis(1);
    assert!(!ticket.owned(&cuts, &b));
    let mut polls = 0;
    assert!(ticket
        .alive(&cuts, &b, &mut || {
            polls += 1;
            true
        })
        .is_none());
    assert_eq!(polls, 0);
    assert!(ticket.begin(&cuts, &b).is_none());
}

#[test]
fn root_coupling_work_payload_or_check_refusal_keeps_preflight_and_old_holds() {
    let p = source(-0.25);
    for refusal in 0..3 {
        let (cuts, plan, mut b, checked, cost) = baseline(&p);
        let old_witness = witness(&plan);
        let old_lower = checked.root_lower.to_bits();
        match refusal {
            0 => b.charge(b.remaining().unwrap() - PREFLIGHT).unwrap(),
            1 => b.admitted_bytes = MAX_BYTES,
            _ => b.state.lock().unwrap().checks = MAX_CHECKS,
        }
        let before = b.state.lock().unwrap().work;
        let checks = b.state.lock().unwrap().checks;
        let reserved = b.state.lock().unwrap().reserved;
        let bytes = b.admitted_bytes;
        assert!(admit(&cuts, &plan, cost, &mut b).is_none());
        assert!(b.root_coupling.is_none());
        let c = b.state.lock().unwrap();
        assert_eq!(c.work, before + PREFLIGHT);
        assert_eq!(c.checks, checks);
        assert_eq!(c.reserved, reserved);
        assert!(c.root_coupling.requested && !c.root_coupling.admitted);
        assert_eq!(c.root_coupling.used_work_units, PREFLIGHT);
        assert_eq!(b.admitted_bytes, bytes);
        assert_eq!(witness(&plan), old_witness);
        assert_eq!(checked.root_lower.to_bits(), old_lower);
    }
}

#[test]
fn root_coupling_baseline_plan_and_cost_seal_rejects_changed_same_shape_bits() {
    let p = source(-0.25);
    for change in 0..5 {
        let (cuts, plan, mut b, _, cost) = baseline(&p);
        let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
        assert!(ticket.matches_plan(&plan, cost));
        let mut changed = plan.clone();
        let mut changed_cost = cost;
        match change {
            0 => changed.components[0].alpha[0] = 0.5_f64.next_up(),
            1 => changed.components[0].g[0] = -0.,
            2 => changed.components[0].positive_units[0] = 2,
            3 => changed.components.swap(0, 1),
            _ => changed_cost += 1,
        }
        assert!(!ticket.matches_plan(&changed, changed_cost));
        ticket.begin(&cuts, &b).unwrap();
        let mut polls = 0;
        assert!(
            resources::choose(&cuts, &changed, &b, &ticket, changed_cost, &mut || {
                polls += 1;
                true
            })
            .is_none()
        );
        assert_eq!(polls, 0);
        assert_eq!(b.state.lock().unwrap().root_coupling.producer_attempts, 1);
        assert_eq!(b.state.lock().unwrap().root_coupling.support_attempts, 0);
    }
}

#[test]
fn root_coupling_last_mask_and_final_barrier_cancellation_keep_checked_baseline() {
    let p = source(-0.25);
    for final_barrier in [false, true] {
        let (cuts, mut plan, mut b, mut checked, mut cost) = baseline(&p);
        let original = witness(&plan);
        let old = (
            checked.root_lower.to_bits(),
            checked.constant.to_bits(),
            cost,
        );
        let work = b.state.lock().unwrap().work;
        let checks = b.state.lock().unwrap().checks;
        let masks = b.state.lock().unwrap().checked_masks;
        let state = b.state.clone();
        let mut refused = false;
        let mut poll = || {
            let c = state.lock().unwrap();
            let at_boundary = if final_barrier {
                c.root_coupling.completed_supports == 1
            } else {
                c.root_coupling.support_attempts == 1 && c.checked_masks == masks + 3
            };
            if at_boundary && !refused {
                refused = true;
                false
            } else {
                true
            }
        };
        assert!(run(
            &cuts,
            &mut plan,
            &mut checked,
            &mut cost,
            cuts.checked_lower,
            &mut b,
            &mut poll
        ));
        assert!(refused);
        assert_eq!(witness(&plan), original);
        assert_eq!(
            (
                checked.root_lower.to_bits(),
                checked.constant.to_bits(),
                cost
            ),
            old
        );
        let c = b.state.lock().unwrap();
        assert_eq!(c.work, work + LIMIT);
        assert_eq!(c.checks, checks + 1);
        assert_eq!(c.root_coupling.support_attempts, 1);
        assert_eq!(
            c.root_coupling.completed_supports,
            usize::from(final_barrier)
        );
        assert!(!c.root_coupling.selected);
        assert_eq!(
            c.root_coupling.stop_reason,
            if final_barrier {
                "completion-barrier"
            } else {
                "whole-source-refused"
            }
        );
        drop(c);
        assert!(!run(
            &cuts,
            &mut plan,
            &mut checked,
            &mut cost,
            cuts.checked_lower,
            &mut b,
            &mut || true
        ));
        assert_eq!(b.state.lock().unwrap().work, work + LIMIT);
    }
}

#[test]
fn root_coupling_selected_owner_keeps_both_publication_holds_after_all_free_work() {
    let p = source(-0.25);
    let (cuts, mut plan, mut b, mut checked, mut cost) = baseline(&p);
    let initial = checked.root_lower;
    // The prepaid eighth checker remains usable after normal slots are exhausted.
    b.state.lock().unwrap().checks = MAX_CHECKS - 1;
    assert!(run(
        &cuts,
        &mut plan,
        &mut checked,
        &mut cost,
        cuts.checked_lower,
        &mut b,
        &mut || true
    ));
    assert!(b.state.lock().unwrap().root_coupling.selected);
    assert_eq!(b.state.lock().unwrap().checks, MAX_CHECKS);
    assert!(checked.root_lower > initial);
    b.trim_unspent_reservation(cost).unwrap();
    component_rows::admit(&cuts, &plan, &b).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    let owner = row(&cuts, &plan, checked, cost, &b, &seal);
    b.charge(b.remaining().unwrap()).unwrap();
    let checks = b.state.lock().unwrap().checks;
    for _ in 0..2 {
        assert_eq!(
            owner
                .replay_inner(&cuts, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            owner.lower.to_bits()
        );
    }
    assert!(owner
        .replay_inner(&cuts, end(), &mut || true, false)
        .is_none());
    let c = b.state.lock().unwrap();
    assert_eq!(c.work, MAX_WORK);
    assert_eq!(c.reserved, 0);
    assert_eq!(c.component_reserved, 0);
    assert_eq!(c.source_replays, 2);
    assert_eq!(c.checks, checks);
    assert_eq!(c.root_coupling.checker_slots, 1);
}

#[test]
fn root_coupling_worker_advances_checked_export_owner_below_stronger_old_floor() {
    let p = source(-0.25);
    let (mut cuts, _, _, _, _) = baseline(&p);
    // Supply a separately Source-checked older floor; no default discovery claim.
    let full = whole();
    let old_budget = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let old_cost = full.cost(&p).unwrap();
    old_budget.reserve(old_cost).unwrap();
    old_budget.proof(old_cost, false).unwrap();
    let old_floor = certificate::check(&cuts, &full, &old_budget, end(), &mut || true)
        .unwrap()
        .root_lower;
    assert!(old_floor < 0. && old_floor > -0.251);
    cuts.checked_lower = old_floor;
    let inspection = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let initial_plan = super::super::geometry::plan(&cuts, &inspection).unwrap();
    let initial_cost = initial_plan.cost(&p).unwrap();
    inspection.reserve(initial_cost).unwrap();
    inspection.proof(initial_cost, false).unwrap();
    let initial = certificate::check(&cuts, &initial_plan, &inspection, end(), &mut || true)
        .unwrap()
        .root_lower;
    assert!(initial < old_floor - 0.1);
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let (owner, report) =
        propose_with_policy(&cuts, &mut b, prepaid::Policy::RootCoupling, &mut || true);
    let owner = owner.unwrap();
    assert!(report.root_coupling.admitted);
    assert!(!report.root_coupling.selected);
    assert!(owner.lower > initial && owner.lower <= old_floor);
    assert_eq!(owner.lower.to_bits(), report.lower.unwrap().to_bits());
    assert_eq!(owner.witness.as_slice(), witness(&owner.plan).as_slice());
    assert!(owner.budget.state.lock().unwrap().reserved >= 2 * owner.cost);
    for _ in 0..2 {
        assert_eq!(
            owner
                .replay_inner(&cuts, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            owner.lower.to_bits()
        );
    }
    assert!(owner
        .replay_inner(&cuts, end(), &mut || true, false)
        .is_none());
    assert_eq!(cuts.checked_lower.to_bits(), old_floor.to_bits());
    assert_eq!(cuts.proposal_lp_calls, 0);
}
