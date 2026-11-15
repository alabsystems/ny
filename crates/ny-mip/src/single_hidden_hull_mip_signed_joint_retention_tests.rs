// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored once-only work, original binding and complete-mask refusal controls.
use super::test_fixture::*;
use super::*;

#[test]
fn signed_joint_no_free_work_or_payload_preserves_old_rows_and_second_publication() {
    for payload in [false, true] {
        let f = Fixture::new(0.25);
        let p = f.source();
        let mut c = cuts(&p, 1);
        if payload {
            let row = c.signed.as_mut().unwrap();
            let bytes = row
                .budget
                .state
                .lock()
                .unwrap()
                .component_rows
                .admitted_bytes;
            row.budget.admitted_bytes = MAX_BYTES - bytes;
        } else {
            let row = c.signed.as_ref().unwrap();
            let free = row.budget.remaining().unwrap();
            row.budget.charge(free).unwrap();
        }
        let mut parts = f.parts();
        let enc = f.encoding(&parts);
        let original = parts.problem.clone();
        let report = c.append_to(&mut parts, &enc, end()).unwrap();
        let row = c.signed.as_ref().unwrap();
        let s = report.signed.unwrap();
        assert_eq!(s.component_rows.appended_rows, 2);
        assert_eq!(s.component_rows.joint.appended_rows, 0);
        assert_eq!(s.component_rows.joint.work_units, 0);
        assert_eq!(
            &parts.problem.rows()[..original.num_rows()],
            original.rows()
        );
        let first = row.stats();
        assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
        assert_eq!(row.stats().source_replays, 2);
        assert_eq!(row.stats().component_rows.joint, first.component_rows.joint);
        assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
        assert!(row.stats().work_units <= MAX_WORK);
    }
}
#[test]
fn signed_joint_exhausted_existing_check_slots_refuse_before_new_scan_or_debit() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    row.budget.state.lock().unwrap().checks = MAX_CHECKS;
    let mut parts = f.parts();
    let enc = f.encoding(&parts);
    let report = c.append_to(&mut parts, &enc, end()).unwrap();
    let r = &report.signed.as_ref().unwrap().component_rows.joint;
    assert_eq!(r.stop_reason, "check-slots");
    assert_eq!(r.checks_before, MAX_CHECKS);
    assert_eq!(r.remaining_checks_before, 0);
    assert_eq!(r.checker_slots, 0);
    assert_eq!(r.work_units, 0);
    assert_eq!(r.checked_masks, 0);
    assert_eq!(r.appended_rows, 0);
    assert_eq!(row.budget.state.lock().unwrap().checks, MAX_CHECKS);
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
    assert_eq!(row.stats().source_replays, 2);
}
#[test]
fn signed_joint_transient_final_mask_refusal_keeps_all_prepared_old_rows_without_new_attempt() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    let parts = f.parts();
    let enc = f.encoding(&parts);
    let mut refused = false;
    let rows = binding::prepare_rows(&c, &parts, &enc, end(), &mut || {
        if !refused && row.stats().component_rows.joint.checked_masks == 3 {
            refused = true;
            false
        } else {
            true
        }
    })
    .unwrap();
    assert!(refused);
    assert_ne!((rows).len(), 0);
    let r = row.stats();
    assert_eq!(r.source_replays, 1);
    assert_eq!(r.component_rows.prepared_rows, 2);
    assert!(r.component_rows.phase.prepared_rows > 0);
    assert_eq!(r.component_rows.joint.checked_masks, 3);
    assert_eq!(r.component_rows.joint.prepared_rows, 0);
    assert_eq!(r.component_rows.joint.producer_attempts, 1);
    assert_eq!(r.component_rows.joint.checker_slots, 1);
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert_eq!(
            (prepare(row, &c, &parts, binding, end(), &mut || true)).len(),
            0
        );
    })
    .unwrap();
    let after = row.stats();
    assert_eq!(after.work_units, r.work_units);
    assert_eq!(after.component_rows.joint, r.component_rows.joint);
}
#[test]
fn signed_joint_foreign_original_model_refuses_before_attempt_and_complete_token_cannot_rebind() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let other = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    let parts = f.parts();
    let foreign = f.parts();
    let enc = f.encoding(&parts);
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let before = row.stats();
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert_eq!(
            (prepare(row, &c, &foreign, binding, end(), &mut || true)).len(),
            0
        );
    })
    .unwrap();
    assert_eq!(row.stats().work_units, before.work_units);
    assert!(!row.stats().component_rows.joint.attempted);
    assert!(start(row));
    let (work, bytes) = resources::setup(row, &c).unwrap();
    resources::pay(row, work, bytes, end()).unwrap();
    assert!(witness_matches(row));
    let mut token = checked::produce(row, &c, end(), &mut || true).unwrap();
    assert!(token.owned(row));
    assert!(!token.owned(other.signed.as_ref().unwrap()));
    token.witness = Arc::new((*row.witness).clone());
    assert!(!token.owned(row));
    token.witness = row.witness.clone();
    token.signature = Arc::new((*row.signature).clone());
    assert!(!token.owned(row));
    token.signature = row.signature.clone();
    token.planes[0].lower += 0.25;
    assert!(!token.owned(row));
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert!(checked::export(row, &c, &parts, binding, &token, end(), &mut || true).is_none());
    })
    .unwrap();
}
#[test]
fn signed_joint_original_deadline_and_completed_rows_never_renew_credit_or_checker_slots() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    let parts = f.parts();
    let enc = f.encoding(&parts);
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let before = row.stats();
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert_eq!(
            (prepare(row, &c, &parts, binding, Instant::now(), &mut || true)).len(),
            0
        );
        assert_eq!(
            (prepare(row, &c, &parts, binding, end(), &mut || true)).len(),
            0
        );
    })
    .unwrap();
    let after = row.stats();
    assert_eq!(after.work_units, before.work_units);
    assert_eq!(after.component_rows.joint.producer_attempts, 1);
    assert_eq!(after.component_rows.joint.checker_slots, 0);
    let p = f.source();
    let c = cuts(&p, 1);
    let mut parts = f.parts();
    let enc = f.encoding(&parts);
    c.append_to(&mut parts, &enc, end()).unwrap();
    let row = c.signed.as_ref().unwrap();
    let first = row.stats();
    assert_eq!(first.component_rows.joint.appended_rows, 1);
    assert!(row.replay_with_cache(&c, end(), &mut || true).is_some());
    let second = row.stats();
    assert_eq!(second.component_rows.joint, first.component_rows.joint);
    assert_eq!(second.source_replays, 2);
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.stats().work_units, second.work_units);
}
