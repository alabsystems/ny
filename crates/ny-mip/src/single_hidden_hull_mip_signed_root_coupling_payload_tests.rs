// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual retained-capacity and once-paid boundary controls.
use super::super::test_fixture::{baseline, source};
use super::*;

#[test]
fn root_payload_streamed_bits_match_old_witness_and_preserve_signed_zero() {
    let p = source(-0.25);
    let (_, plan, _, _, _) = baseline(&p);
    let expected = witness(&plan);
    let sealed = seal(&plan).unwrap();
    assert_eq!(sealed.as_ref(), expected.as_slice());
    assert!(matches(&plan, &sealed));
    for field in 0..5 {
        let mut changed = plan.clone();
        match field {
            0 => changed.components[0].g[0] = -0.,
            1 => changed.components[0].alpha[0] = 0.25,
            2 => changed.components[0].positive_units[0] = 2,
            3 => changed.components[0].negative[0] = 2,
            _ => changed.leftover.push(0),
        }
        assert!(!matches(&changed, &sealed));
        assert_ne!(witness(&changed), expected);
    }
    assert!(!matches(&plan, &sealed[..sealed.len() - 1]));
    let mut extra = sealed.to_vec();
    extra.push(0);
    assert!(!matches(&plan, &extra));
}

#[test]
fn root_payload_unused_capacity_is_counted_and_refused_before_full_debit() {
    for field in 0..7 {
        let p = source(-0.25);
        let (cuts, mut plan, mut b, _, cost) = baseline(&p);
        let (_, _, new_words) = resources::admission(&plan, &b).unwrap();
        let ceiling = plan_bound(new_words, plan.components.len()).unwrap();
        let old_values = witness(&plan);
        let old_heap = heap(&plan).unwrap();
        match field {
            0 => plan
                .components
                .reserve_exact(ceiling / size_of::<Component>() + 1),
            1 => plan
                .leftover
                .reserve_exact(ceiling / size_of::<usize>() + 1),
            2 => plan.components[0]
                .negative
                .reserve_exact(ceiling / size_of::<usize>() + 1),
            3 => plan.components[0]
                .positive
                .reserve_exact(ceiling / size_of::<usize>() + 1),
            4 => plan.components[0].positive_units.reserve_exact(ceiling + 1),
            5 => plan.components[0]
                .g
                .reserve_exact(ceiling / size_of::<f64>() + 1),
            _ => plan.components[0]
                .alpha
                .reserve_exact(ceiling / size_of::<f64>() + 1),
        }
        assert_eq!(witness(&plan), old_values);
        assert!(heap(&plan).unwrap() > ceiling && heap(&plan).unwrap() > old_heap);
        let work = b.state.lock().unwrap().work;
        let bytes = b.admitted_bytes;
        let checks = b.state.lock().unwrap().checks;
        assert!(admit(&cuts, &plan, cost, &mut b).is_none());
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.work, work + PREFLIGHT);
        assert_eq!(counts.checks, checks);
        assert_eq!(b.admitted_bytes, bytes);
        assert!(!counts.root_coupling.admitted);
        assert!(counts.root_coupling.requested);
    }
}

#[test]
fn root_payload_capacity_change_after_admission_cannot_reenter_source_work() {
    let p = source(-0.25);
    let (cuts, mut plan, mut b, _, cost) = baseline(&p);
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    let before = witness(&plan);
    let capacity = heap(&plan).unwrap();
    assert!(ticket.matches_plan(&plan, cost));
    plan.leftover.reserve_exact(1);
    assert_eq!(witness(&plan), before);
    assert_ne!(heap(&plan), Some(capacity));
    assert!(!ticket.matches_plan(&plan, cost));
    ticket.begin(&cuts, &b).unwrap();
    let work = b.state.lock().unwrap().work;
    let bytes = b.admitted_bytes;
    let mut polls = 0;
    assert!(resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert_eq!(b.state.lock().unwrap().work, work);
    assert_eq!(b.admitted_bytes, bytes);
    assert!(resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).is_none());
}

#[test]
fn root_payload_exact_cap_admits_and_one_more_byte_refuses_without_refund() {
    for extra in [0, 1] {
        let p = source(-0.25);
        let (cuts, plan, mut b, _, cost) = baseline(&p);
        let (_, payload, _) = resources::admission(&plan, &b).unwrap();
        b.admitted_bytes = MAX_BYTES - payload + extra;
        let initial_bytes = b.admitted_bytes;
        let work = b.state.lock().unwrap().work;
        let checks = b.state.lock().unwrap().checks;
        let ticket = admit(&cuts, &plan, cost, &mut b);
        if extra == 0 {
            let (ticket, _) = ticket.unwrap();
            assert_eq!(b.admitted_bytes, MAX_BYTES);
            assert_eq!(b.state.lock().unwrap().work, work + LIMIT);
            assert_eq!(b.state.lock().unwrap().checks, checks + 1);
            assert_eq!(ticket.payload, payload);
            ticket.begin(&cuts, &b).unwrap();
            assert!(resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || false).is_none());
            assert_eq!(b.admitted_bytes, MAX_BYTES);
            assert_eq!(b.state.lock().unwrap().work, work + LIMIT);
            assert!(admit(&cuts, &plan, cost, &mut b).is_none());
        } else {
            assert!(ticket.is_none());
            assert_eq!(b.admitted_bytes, initial_bytes);
            assert_eq!(b.state.lock().unwrap().work, work + PREFLIGHT);
            assert_eq!(b.state.lock().unwrap().checks, checks);
        }
    }
}

#[test]
fn root_payload_real_geometry_owner_witness_peak_fits_the_prepaid_bound() {
    let p = source(-0.25);
    let (cuts, plan, mut b, _, cost) = baseline(&p);
    let (_, payload, new_words) = resources::admission(&plan, &b).unwrap();
    let ceiling = plan_bound(new_words, plan.components.len()).unwrap();
    let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
    assert_eq!(ticket.baseline_seal.as_ref(), witness(&plan).as_slice());
    ticket.begin(&cuts, &b).unwrap();
    let choice = resources::choose(&cuts, &plan, &b, &ticket, cost, &mut || true).unwrap();
    ticket.pay(&cuts, &b, choice.total).unwrap();
    let mut built = geometry::build(&cuts, &plan, &choice, &b, &ticket, &mut || true).unwrap();
    assert!(heap(&built.plan).unwrap() <= ceiling);
    let selected = built.selected;
    steering::optimize(
        &cuts,
        &mut built.plan.components[selected],
        choice.active,
        SearchRole::Selected,
        &b,
        &ticket,
        &mut || true,
    )
    .unwrap();
    ticket.claim_checker(&cuts, &b).unwrap();
    let checked = certificate::check(&cuts, &built.plan, &b, ticket.end, &mut || true).unwrap();
    // Model replacement of the old working Plan before constructing an owner.
    drop(plan);
    let shared_source_signature = Arc::new(signature(&cuts).collect());
    let owner = row(
        &cuts,
        &built.plan,
        checked,
        choice.check,
        &b,
        &shared_source_signature,
    );
    assert!(Arc::ptr_eq(&owner.signature, &shared_source_signature));
    let third_plan = built.plan.clone();
    let second_witness = witness(&third_plan);
    let actual = heap(&built.plan).unwrap()
        + heap(&owner.plan).unwrap()
        + heap(&third_plan).unwrap()
        + owner.witness.capacity() * size_of::<u64>()
        + second_witness.capacity() * size_of::<u64>();
    assert!(actual <= 9 * ceiling);
    assert!(ticket.baseline_seal.len() * size_of::<u64>() + actual <= payload);
    assert_eq!(owner.witness.as_slice(), witness(&owner.plan).as_slice());
    assert_eq!(b.state.lock().unwrap().replay_attempts, 0);
}
