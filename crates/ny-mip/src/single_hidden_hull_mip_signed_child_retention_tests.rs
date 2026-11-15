// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Refusals retain previous authority; no extra mask pass or renewed quota.
use super::super::tests::{cuts, end};
use super::tests::{cached, checked_row, one, plan, split};
use super::*;

#[test]
fn signed_child_last_mask_and_final_pass_refusal_never_mint_partial_cache() {
    for additional in [1, 2] {
        let p = one(0.25);
        let c = cuts(&p);
        let row = checked_row(&c, &plan(&p));
        row.replay_inner(&c, end(), &mut || true, false).unwrap();
        let before = row.budget.state.lock().unwrap().checked_masks;
        let stop = before + additional;
        let mut poll = || row.budget.state.lock().unwrap().checked_masks < stop;
        assert!(row.replay_with_cache(&c, end(), &mut poll).is_none());
        let state = row.budget.state.lock().unwrap();
        assert_eq!(state.checked_masks, stop);
        assert_eq!(state.source_replays, 1);
        assert_eq!(state.replay_attempts, 2);
        assert_eq!(state.child.collection_attempts, 1);
        assert_eq!(state.child.cached_masks, 0);
        drop(state);
        assert!(row.replay_with_cache(&c, end(), &mut || true).is_none());
    }
}

#[test]
fn signed_child_foreign_source_same_address_edits_and_old_tuple_bits_refuse_binding() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let foreign = p.clone();
    assert!(cache.bind(&cuts(&foreign), end(), &mut || true).is_none());
    for kind in 0..3 {
        let mut c = cuts(&p);
        match kind {
            0 => c.multipliers.g[0] += 0.125,
            1 => c.multipliers.alpha[0] = 0.25,
            _ => c.checked_lower += 0.125,
        }
        assert!(cache.bind(&c, end(), &mut || true).is_none());
    }
    for kind in 0..4 {
        let mut p = one(0.25);
        let cache = {
            let c = cuts(&p);
            let row = checked_row(&c, &plan(&p));
            cached(&c, &row)
        };
        match kind {
            0 => p.w1[0] += 0.125,
            1 => p.b1[0] += 0.125,
            2 => p.lower[0] = -0.5,
            _ => p.d += 0.125,
        }
        assert!(cache.bind(&cuts(&p), end(), &mut || true).is_none());
    }
    assert_eq!(cache.stats().completed_calls, 0);
}

#[test]
fn signed_child_retained_payload_and_head_map_seal_refuse_each_mutated_cache_bit() {
    for kind in 0..7 {
        let p = one(0.25);
        let c = cuts(&p);
        let row = checked_row(&c, &plan(&p));
        let mut cache = cached(&c, &row);
        let v = Arc::get_mut(&mut cache).unwrap();
        match kind {
            0 => v.tables[0].values[1] += 0.125,
            1 => v.tables[0].minimum += 0.125,
            2 => v.tables[0].heads[0] = 99,
            3 => v.head_slot[0] = None,
            4 => v.g[0] += 0.125,
            5 => v.constant += 0.125,
            _ => v.root_lower += 0.125,
        }
        assert!(cache.bind(&c, end(), &mut || true).is_none());
        assert_eq!(cache.stats().completed_calls, 0);
    }
}

#[test]
fn signed_child_original_deadline_and_bound_borrow_cannot_be_renewed_by_later_end() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(bound.lower(&[], &[], expired, &mut || true).is_none());
    assert_eq!(cache.stats().completed_calls, 0);
    let mut cache = cache;
    Arc::get_mut(&mut cache).unwrap().budget.authority = expired;
    assert!(cache
        .bind(&c, end() + Duration::from_secs(60), &mut || true)
        .is_none());
    assert_eq!(cache.stats().stop_reason, "deadline");
    assert_eq!(row.stats().source_replays, 2);
}

#[test]
fn signed_child_shared_call_and_work_caps_never_reset_original_signed_ledger() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    for _ in 0..MAX_CHILD_CALLS {
        assert!(bound.lower(&[], &[], end(), &mut || true).is_some());
    }
    assert!(bound.lower(&[], &[], end(), &mut || true).is_none());
    let r = cache.stats();
    assert_eq!(r.attempted_calls, MAX_CHILD_CALLS);
    assert_eq!(r.completed_calls, MAX_CHILD_CALLS);
    assert_eq!(r.stop_reason, "call-cap");
    assert_eq!(row.stats().source_replays, 2);
    assert!(row.stats().work_units <= MAX_WORK);
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let left = row.budget.remaining().unwrap();
    row.budget.charge(left).unwrap();
    assert!(cache.bind(&c, end(), &mut || true).is_none());
    assert_eq!(cache.stats().stop_reason, "work");
    assert_eq!(row.stats().work_units, MAX_WORK);
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(c.proposal_lp_calls, 0);
}

#[test]
fn signed_child_optional_collection_credit_or_payload_refusal_keeps_second_root_replay() {
    for payload in [false, true] {
        let p = one(0.25);
        let c = cuts(&p);
        let mut row = checked_row(&c, &plan(&p));
        row.replay_inner(&c, end(), &mut || true, false).unwrap();
        if payload {
            row.budget.admitted_bytes = MAX_BYTES;
        } else {
            let left = row.budget.remaining().unwrap();
            row.budget.charge(left).unwrap();
        }
        let replay = row.replay_with_cache(&c, end(), &mut || true).unwrap();
        assert!(replay.cache.is_none());
        assert_eq!(replay.root_lower.to_bits(), row.lower.to_bits());
        assert_eq!(row.stats().source_replays, 2);
        assert_eq!(row.budget.state.lock().unwrap().reserved, 0);
        assert_eq!(row.budget.state.lock().unwrap().child.cached_masks, 0);
        assert_eq!(
            row.budget.state.lock().unwrap().child.collection_attempts,
            1
        );
        assert!(row.stats().work_units <= MAX_WORK);
    }
}

#[test]
fn signed_child_incomplete_scan_and_failed_beta_keep_completed_floor_and_count_real_completions() {
    let p = one(0.25);
    let c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    let cache = cached(&c, &row);
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let first = bound
        .lower(&[split(0, false)], &[0.], end(), &mut || true)
        .unwrap();
    assert!(first > 0.249);
    assert!(bound
        .lower(&[split(0, false)], &[0.], end(), &mut || false)
        .is_none());
    assert!(bound
        .lower(&[split(0, false)], &[f64::INFINITY], end(), &mut || true)
        .is_none());
    assert_eq!(cache.stats().attempted_calls, 3);
    assert_eq!(cache.stats().completed_calls, 1);
    let again = bound
        .lower(&[split(0, false)], &[0.], end(), &mut || true)
        .unwrap();
    assert_eq!(again.to_bits(), first.to_bits());
    assert_eq!(cache.stats().completed_calls, 2);
    assert_eq!(cache.stats().positive_calls, 2);
    assert_eq!(row.stats().source_replays, 2);
}

#[test]
fn signed_child_nonpositive_second_replay_cache_survives_later_exhausted_public_root_request() {
    let p = one(0.25);
    let mut c = cuts(&p);
    let row = checked_row(&c, &plan(&p));
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    c.signed = Some(row);
    assert!(c.signed_root_proof(end()).is_none());
    let retained = c.signed_child_cache.get().unwrap().clone();
    assert_eq!(retained.stats().cached_masks, 2);
    assert!(c.signed_root_proof(end()).is_none());
    assert!(Arc::ptr_eq(c.signed_child_cache.get().unwrap(), &retained));
    assert_eq!(c.signed.as_ref().unwrap().stats().source_replays, 2);
    let bound = retained.bind(&c, end(), &mut || true).unwrap();
    assert!(
        bound
            .lower(&[split(0, false)], &[0.], end(), &mut || true)
            .unwrap()
            > 0.249
    );
    assert_eq!(c.proposal_lp_calls, 0);
}
