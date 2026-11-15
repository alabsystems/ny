// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Engineering ownership controls using a supplied, original-Source-checked plan.
//! Default-discovery controls live separately and use the new public entry.
use super::test_fixture::{cuts, end, plan, Fixture};
use super::*;

fn funded(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut c = cuts(p, 1);
    let mut row = c.signed.take().unwrap();
    let mut b = row.budget.clone();
    let before = b.state.lock().unwrap().work;
    let checks = b.state.lock().unwrap().checks;
    let bytes = b.admitted_bytes;
    assert!(prepaid::admit(&c, &row.plan, &mut b));
    assert_eq!(b.state.lock().unwrap().work - before, prepaid::WORK);
    assert_eq!(b.state.lock().unwrap().checks, checks + 1);
    assert_eq!(
        b.admitted_bytes - bytes,
        b.state.lock().unwrap().component_rows.joint.prepaid_bytes
    );
    row.budget = b;
    c.signed = Some(row);
    c
}

#[test]
fn prepaid_joint_shape_payload_bound_and_original_constants_are_explicit() {
    assert_eq!(prepaid::WORK, resources::MAX_JOINT_WORK);
    assert_eq!(MAX_WORK, 64_000_000);
    assert_eq!(MAX_BYTES, 8 << 20);
    assert_eq!((MAX_CHECKS, resources::MAX_PAIRS), (8, 2));
    assert_eq!(prepaid::payload(30, 128, 62, 16), Some(242720));
    assert_eq!(prepaid::payload(128, 512, 256, 64), Some(1010720));
    for (d, h, p, c) in [
        (0, 128, 62, 16),
        (129, 128, 62, 16),
        (30, 513, 62, 16),
        (30, 128, 0, 16),
        (30, 128, 62, 65),
        (1, 2, 1, 3),
        (usize::MAX, 128, 62, 16),
    ] {
        assert!(prepaid::payload(d, h, p, c).is_none());
    }
}

#[test]
fn prepaid_joint_debit_and_payload_survive_consumption_without_refund_or_rebank() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    let c = funded(&p);
    let row = c.signed.as_ref().unwrap();
    let ticket = row.budget.prepaid.as_ref().unwrap();
    let work = row.budget.state.lock().unwrap().work;
    let bytes = row.budget.admitted_bytes;
    let mut copied = row.budget.clone();
    assert!(!prepaid::admit(&c, &row.plan, &mut copied));
    assert_eq!(copied.state.lock().unwrap().work, work);
    assert!(ticket.begin(row, end()));
    assert!(!ticket.begin(row, end()));
    let before = ticket.remaining(row).unwrap();
    ticket.pay(row, 100, 200, end()).unwrap();
    assert_eq!(
        ticket.remaining(row).unwrap(),
        (before.0 - 100, before.1 - 200)
    );
    assert_eq!(row.budget.state.lock().unwrap().work, work);
    assert_eq!(row.budget.admitted_bytes, bytes);
    let used = row
        .budget
        .state
        .lock()
        .unwrap()
        .component_rows
        .joint
        .prepaid_used_work_units;
    assert_eq!(used, 4196);
    assert!(ticket.pay(row, before.0, 0, end()).is_none());
    assert_eq!(row.budget.state.lock().unwrap().work, work);
    assert_eq!(
        ticket.remaining(row).unwrap(),
        (before.0 - 100, before.1 - 200)
    );
}

#[test]
fn prepaid_joint_private_producer_claim_is_once_before_any_repeated_scan() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    let c = funded(&p);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let ticket = row.budget.prepaid.as_ref().unwrap();
    assert!(ticket.begin(row, end()));
    let (work, bytes) = resources::setup(row, &c).unwrap();
    resources::pay(row, work, bytes, end()).unwrap();
    let before = row.budget.state.lock().unwrap().work;
    let checks = row.budget.state.lock().unwrap().checks;
    let first = checked::produce(row, &c, end(), &mut || true).unwrap();
    assert_eq!(first.planes.len(), 1);
    let used = ticket.remaining(row).unwrap();
    let mut polls = 0;
    assert!(checked::produce(row, &c, end(), &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert_eq!(ticket.remaining(row).unwrap(), used);
    assert_eq!(row.budget.state.lock().unwrap().work, before);
    assert_eq!(row.budget.state.lock().unwrap().checks, checks);
    assert!(!ticket.claim_checker(row, end()));
}

#[test]
fn prepaid_joint_prepaid_slot_works_at_eight_and_both_publication_holds_survive() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    let c = funded(&p);
    let row = c.signed.as_ref().unwrap();
    row.budget.state.lock().unwrap().checks = MAX_CHECKS;
    row.budget.charge(row.budget.remaining().unwrap()).unwrap();
    let mut parts = f.parts();
    let encoding = f.encoding(&parts);
    let report = c.append_to(&mut parts, &encoding, end()).unwrap();
    let r = report.signed.unwrap();
    assert_eq!(r.source_replays, 1);
    assert_eq!(r.component_rows.joint.appended_rows, 1);
    assert_eq!(r.component_rows.joint.completed_pairs, 1);
    assert_eq!(r.component_rows.joint.checker_slots, 1);
    assert_eq!(r.component_rows.joint.prepaid_checker_slots, 1);
    assert_eq!(r.component_rows.domination.producer_attempts, 0);
    assert_eq!(row.budget.state.lock().unwrap().checks, MAX_CHECKS);
    let lower = row.lower.to_bits();
    assert_eq!(
        row.replay_inner(&c, end(), &mut || true, false)
            .unwrap()
            .root_lower
            .to_bits(),
        lower
    );
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.budget.state.lock().unwrap().work, MAX_WORK);
    assert_eq!(row.budget.state.lock().unwrap().reserved, 0);
    assert_eq!(row.budget.state.lock().unwrap().source_replays, 2);
}

#[test]
fn prepaid_joint_foreign_ledger_source_dimensions_clocks_and_payload_do_not_own_ticket() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    let mut c = funded(&p);
    let mut row = c.signed.take().unwrap();
    let ticket = row.budget.prepaid.as_ref().unwrap().clone();
    let original = row.budget.clone();
    assert!(ticket.owned(&row));
    row.budget.state = Arc::new(std::sync::Mutex::new(Counts::default()));
    assert!(!ticket.owned(&row));
    row.budget = original.clone();
    row.budget.source ^= 1;
    assert!(!ticket.owned(&row));
    row.budget = original.clone();
    row.budget.dimensions.2 += 1;
    assert!(!ticket.owned(&row));
    row.budget = original.clone();
    row.budget.search_end += Duration::from_secs(1);
    assert!(!ticket.owned(&row));
    row.budget = original.clone();
    row.budget.admitted_bytes -= 1;
    assert!(!ticket.owned(&row));
    row.budget = original;
    assert!(ticket.owned(&row));
    assert!(ticket.begin(&row, end()));
}

#[test]
fn prepaid_joint_expired_and_cancelled_first_attempt_keep_all_debt_and_never_renew() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    let c = funded(&p);
    let row = c.signed.as_ref().unwrap();
    let ticket = row.budget.prepaid.as_ref().unwrap();
    let before = row.budget.state.lock().unwrap().work;
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(!ticket.begin(row, expired));
    let mut polls = 0;
    assert!(checked::produce(row, &c, end(), &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert!(!ticket.begin(row, end()));
    assert_eq!(row.budget.state.lock().unwrap().work, before);
    let c = funded(&p);
    let row = c.signed.as_ref().unwrap();
    let mut parts = f.parts();
    let encoding = f.encoding(&parts);
    let mut poll = || {
        !row.budget
            .state
            .lock()
            .unwrap()
            .component_rows
            .joint
            .attempted
    };
    assert!(c
        .append_with_poll(&mut parts, &encoding, end(), &mut poll)
        .is_none());
    let r = row
        .budget
        .state
        .lock()
        .unwrap()
        .component_rows
        .joint
        .clone();
    assert!(r.prepaid_admitted && r.attempted);
    assert_eq!(r.prepaid_work_units, prepaid::WORK);
    assert_eq!(r.appended_rows, 0);
    assert_eq!(r.paid_pairs, 0);
    assert!(!row.budget.prepaid.as_ref().unwrap().begin(row, end()));
    let work = row.budget.state.lock().unwrap().work;
    let second = c.append_to(&mut parts, &encoding, end()).unwrap();
    let second_joint = second.signed.unwrap().component_rows.joint;
    assert_eq!(second_joint.prepaid_work_units, prepaid::WORK);
    assert_eq!(
        second_joint.prepaid_used_work_units,
        r.prepaid_used_work_units
    );
    assert_eq!(second_joint.paid_pairs, 0);
    assert_eq!(second_joint.checker_slots, 0);
    assert_eq!(second_joint.appended_rows, 0);
    assert_eq!(row.budget.state.lock().unwrap().work, work + row.cost);
    assert_eq!(row.budget.state.lock().unwrap().source_replays, 2);
    assert!(c.append_to(&mut parts, &encoding, end()).is_none());
}

#[test]
fn prepaid_joint_refused_work_or_payload_keeps_only_preflight_debt_and_old_owner() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    for payload in [false, true] {
        let mut c = cuts(&p, 1);
        let row = c.signed.take().unwrap();
        let mut b = row.budget.clone();
        if payload {
            b.admitted_bytes = MAX_BYTES;
        } else {
            b.charge(b.remaining().unwrap() - 4096).unwrap();
        }
        let before = b.state.lock().unwrap().work;
        let checks = b.state.lock().unwrap().checks;
        let reserved = b.state.lock().unwrap().reserved;
        assert!(!prepaid::admit(&c, &row.plan, &mut b));
        assert!(b.prepaid.is_none());
        assert_eq!(b.state.lock().unwrap().work, before + 4096);
        assert_eq!(b.state.lock().unwrap().checks, checks);
        assert_eq!(b.state.lock().unwrap().reserved, reserved);
        assert_eq!(
            b.state
                .lock()
                .unwrap()
                .component_rows
                .joint
                .prepaid_work_units,
            4096
        );
        // Refusal does not replace the checked legacy witness or its ledger.
        assert!(Arc::ptr_eq(&row.budget.state, &b.state));
        assert_eq!(row.witness.as_slice(), witness(&plan(&p, 1)).as_slice());
    }
}

#[test]
fn prepaid_joint_nonimproving_old_floor_retains_and_advances_checked_export_owner() {
    let f = Fixture::new(-0.25);
    let p = f.source();
    let mut c = cuts(&p, 1);
    c.signed = None;
    // A separately checked whole-mask certificate supplies a stronger OLD floor.
    // This engineering fixture does not claim default partition discovery.
    let whole = Plan {
        components: vec![Component {
            negative: vec![1, 2],
            positive: vec![0],
            positive_units: vec![2],
            g: vec![0.],
            alpha: vec![0., 0.25, 0.75, 1.],
        }],
        leftover: vec![],
    };
    let old_budget = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let old_cost = whole.cost(&p).unwrap();
    old_budget.reserve(old_cost).unwrap();
    old_budget.proof(old_cost, false).unwrap();
    let old_floor = certificate::check(&c, &whole, &old_budget, end(), &mut || true)
        .unwrap()
        .root_lower;
    assert!(old_floor < 0. && old_floor > -0.251);
    c.checked_lower = old_floor;
    let inspection = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let initial_plan = geometry::plan(&c, &inspection).unwrap();
    let initial_cost = initial_plan.cost(&p).unwrap();
    inspection.reserve(initial_cost).unwrap();
    inspection.proof(initial_cost, false).unwrap();
    let initial = certificate::check(&c, &initial_plan, &inspection, end(), &mut || true)
        .unwrap()
        .root_lower;
    assert!(initial < old_floor - 0.1);
    let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let (owner, report) = propose_with_policy(&c, &mut b, prepaid::Policy::OneJoint, &mut || true);
    let owner = owner.unwrap();
    assert!(report.component_rows.joint.prepaid_admitted);
    assert!(owner.lower > initial && owner.lower <= old_floor);
    assert_eq!(owner.lower.to_bits(), report.lower.unwrap().to_bits());
    assert_eq!(owner.witness.as_slice(), witness(&owner.plan).as_slice());
    assert!(owner.budget.state.lock().unwrap().reserved >= 2 * owner.cost);
    for _ in 0..2 {
        assert_eq!(
            owner
                .replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            owner.lower.to_bits()
        );
    }
    assert_eq!(c.checked_lower.to_bits(), old_floor.to_bits());
    assert_eq!(c.proposal_lp_calls, 0);
}
