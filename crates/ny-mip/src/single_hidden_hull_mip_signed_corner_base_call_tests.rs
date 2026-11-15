// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine completed-cache controls. Padding explicitly consumes unused work;
//! it creates no Source, publication, bind, or numerical authority.
use super::*;
use crate::single_hidden_hull::bab::HullSplit;

fn fill_remaining(b: &Budget) -> (usize, usize, usize, usize) {
    let mut c = b.state.lock().unwrap();
    c.work = MAX_WORK - c.reserved - c.component_reserved;
    (c.work, c.child.work_units, c.reserved, c.component_reserved)
}

fn unchanged_debit(b: &Budget, paid: (usize, usize, usize, usize)) {
    let c = b.state.lock().unwrap();
    assert_eq!(
        (c.work, c.child.work_units, c.reserved, c.component_reserved),
        paid
    );
}

#[test]
fn corner_prepaid_base_calls_reach_all_sixteen_prices_without_spare_work() {
    assert_eq!(base_call_cap(30, 128, 1308, 16), Some(62_816));
    assert_eq!(16 * base_call_cap(30, 128, 1308, 16).unwrap(), 1_005_056);
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let base_bytes = b.admitted_bytes;
    let cache = completed(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let paid = fill_remaining(b);
    for i in 0..16 {
        assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() > 0.13);
        unchanged_debit(b, paid);
        let c = b.state.lock().unwrap();
        let grant = c.corner.grant.as_ref().unwrap();
        assert_eq!((grant.base_calls, grant.prices), (15 - i, 15 - i));
        assert_eq!(c.child.completed_calls, i + 1);
    }
    assert!(bound.lower_zero(&[], end(), &mut || true).is_none());
    assert!(bound.root_lower().unwrap() > 0.13);
    unchanged_debit(b, paid);
    let c = b.state.lock().unwrap();
    assert_eq!((c.child.attempted_calls, c.child.completed_calls), (17, 16));
    assert_eq!(c.child.stop_reason, "work");
    assert_eq!(b.admitted_bytes, base_bytes);
    assert_eq!(cuts.proposal_lp_calls, 1);
}

#[test]
fn corner_prepaid_base_deadline_and_malformed_attempts_burn_without_refund() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let cache = completed(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let paid = fill_remaining(b);
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(bound.lower_zero(&[], expired, &mut || true).is_none());
    let split = HullSplit {
        neuron: 0,
        active: true,
    };
    assert!(bound
        .lower(&[split], &[f64::INFINITY], end(), &mut || true)
        .is_none());
    assert!(bound
        .lower_zero(&[split, split], end(), &mut || true)
        .is_none());
    assert!(bound
        .lower_zero(
            &[HullSplit {
                neuron: u32::MAX,
                active: false
            }],
            end(),
            &mut || true
        )
        .is_none());
    // An invalid top-level shape is refused before the existing charge seam.
    assert!(bound.lower(&[], &[0.], end(), &mut || true).is_none());
    {
        let c = b.state.lock().unwrap();
        let grant = c.corner.grant.as_ref().unwrap();
        assert_eq!((grant.base_calls, grant.prices), (12, 16));
        assert_eq!((c.child.attempted_calls, c.child.completed_calls), (4, 0));
    }
    assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() > 0.13);
    unchanged_debit(b, paid);
    let c = b.state.lock().unwrap();
    let grant = c.corner.grant.as_ref().unwrap();
    assert_eq!((grant.base_calls, grant.prices), (11, 15));
    assert_eq!((c.child.attempted_calls, c.child.completed_calls), (5, 1));
}

#[test]
fn corner_prepaid_base_over_cap_uses_original_debit_and_preserves_credit() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let cache = completed(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let (cap, before) = {
        let c = b.state.lock().unwrap();
        (
            c.corner.grant.as_ref().unwrap().base_call_cap,
            (c.work, c.child.work_units),
        )
    };
    // Explicit ledger boundary probe: this fixture has only six heads, so a
    // real seventeen-split query would fail shape before reaching this seam.
    b.child_charge(cap + 1, true, end()).unwrap();
    {
        let c = b.state.lock().unwrap();
        assert_eq!(
            (c.work, c.child.work_units),
            (before.0 + cap + 1, before.1 + cap + 1)
        );
        assert_eq!(c.corner.grant.as_ref().unwrap().base_calls, 16);
    }
    let paid = fill_remaining(b);
    assert!(b.child_charge(cap + 1, true, end()).is_none());
    assert!(b.child_charge(usize::MAX, true, end()).is_none());
    assert_eq!(
        b.state
            .lock()
            .unwrap()
            .corner
            .grant
            .as_ref()
            .unwrap()
            .base_calls,
        16
    );
    assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() > 0.13);
    unchanged_debit(b, paid);
    assert_eq!(
        b.state
            .lock()
            .unwrap()
            .corner
            .grant
            .as_ref()
            .unwrap()
            .base_calls,
        15
    );
}

#[test]
fn corner_prepaid_base_clones_share_credits_and_keep_original_call_cap() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let cache = completed(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let cap = b
        .state
        .lock()
        .unwrap()
        .corner
        .grant
        .as_ref()
        .unwrap()
        .base_call_cap;
    let alias: Budget = b.clone();
    let paid = fill_remaining(b);
    alias.child_charge(cap, true, end()).unwrap();
    assert_eq!(
        b.state
            .lock()
            .unwrap()
            .corner
            .grant
            .as_ref()
            .unwrap()
            .base_calls,
        15
    );
    assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() > 0.13);
    assert_eq!(
        alias
            .state
            .lock()
            .unwrap()
            .corner
            .grant
            .as_ref()
            .unwrap()
            .base_calls,
        14
    );
    // The independent old attempt ceiling remains authoritative even while
    // eligible prepaid work exists. This counter saturation is test-only.
    b.state.lock().unwrap().child.attempted_calls = child::MAX_CHILD_CALLS;
    assert!(bound.lower_zero(&[], end(), &mut || true).is_none());
    let c = b.state.lock().unwrap();
    assert_eq!(c.corner.grant.as_ref().unwrap().base_calls, 14);
    assert_eq!(c.child.stop_reason, "call-cap");
    assert_eq!(c.child.attempted_calls, child::MAX_CHILD_CALLS);
    drop(c);
    unchanged_debit(b, paid);
}

#[test]
fn corner_prepaid_base_calls_remain_usable_when_optional_binding_refuses() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let mut cache = completed(&cuts);
    child::corner_extension_for_test(Arc::get_mut(&mut cache).unwrap())
        .as_mut()
        .unwrap()
        .seal[0] ^= 1;
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    assert!(bound.root_lower().unwrap() < -1.88);
    let paid = fill_remaining(b);
    for _ in 0..16 {
        assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() < -1.88);
    }
    assert!(bound.lower_zero(&[], end(), &mut || true).is_none());
    unchanged_debit(b, paid);
    let c = b.state.lock().unwrap();
    let grant = c.corner.grant.as_ref().unwrap();
    assert_eq!((grant.base_calls, grant.prices), (0, 16));
    assert_eq!((c.child.attempted_calls, c.child.completed_calls), (17, 16));
    assert_eq!(c.child.stop_reason, "work");
}
