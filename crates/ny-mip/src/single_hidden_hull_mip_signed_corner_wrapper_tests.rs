// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Admission, authority and fallback controls for the genuine cached fixture.
//! Real Source/capture/native/child funding precedes every optional admission.
//! Resource padding and sidecar corruption are explicitly test-only controls.
use super::*;

fn holds(b: &Budget) -> (usize, usize, usize, usize) {
    let c = b.state.lock().unwrap();
    (
        c.reserved,
        c.component_reserved,
        c.prepaid_collection_work,
        c.prepaid_bind_work,
    )
}
fn quote_for(f: &CrossFixture, k: usize) -> (usize, usize) {
    let plan = f.plan();
    let bits: Vec<_> = plan
        .components
        .iter()
        .map(|c| c.negative.len() as u8)
        .collect();
    let q = quote::quote(CrossFixture::D, &bits, k).unwrap();
    let masks: usize = bits.iter().map(|&n| 1usize << n).sum();
    let entry = quote::CHILD_PRICES
        * base_call_cap(CrossFixture::D, f.dimensions[1], masks, bits.len()).unwrap();
    (
        SETUP_WORK + q.work + 16_384 + 32 * masks + 128 * bits.len() + entry,
        q.bytes + WRAPPER_BYTES,
    )
}
fn completed(cuts: &HullRootMipCuts<'_>) -> Arc<child::Cache> {
    let row = cuts.signed.as_ref().unwrap();
    let first = row.replay(cuts, end(), &mut || true).unwrap();
    assert!(first.cache.is_none());
    let second = row.replay_with_cache(cuts, end(), &mut || true).unwrap();
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(second.root_lower.to_bits(), row.lower.to_bits());
    second.cache.unwrap()
}

fn admission_edge(work_edge: bool, k: usize, excess: usize) {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, false);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let original_holds = holds(b);
    let base = b.admitted_bytes;
    let (cost, payload) = quote_for(&f, k);
    let (old_work, old_child_work, old_cache_bytes) = {
        let mut c = b.state.lock().unwrap();
        if work_edge {
            let target = MAX_WORK - c.reserved - c.component_reserved - cost + excess;
            assert!(
                target >= c.work,
                "fixture must have the requested spare work"
            );
            c.work = target; // Explicit synthetic consumption of unused work.
        } else {
            let total = base
                + c.shared_bytes
                + c.direction.cache_bytes
                + c.child.cache_bytes
                + c.component_rows.admitted_bytes;
            let target = MAX_BYTES - payload + excess;
            assert!(
                target >= total,
                "fixture must have the requested spare bytes"
            );
            c.component_rows.admitted_bytes += target - total;
        }
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    let result = prepay(&cuts, &f.plan(), b);
    let expected = if excess == 0 {
        Some(k)
    } else if k > 1 {
        Some(k - 1)
    } else {
        None
    };
    assert_eq!(result.is_some(), expected.is_some());
    assert_eq!(holds(b), original_holds);
    assert_eq!(b.admitted_bytes, base);
    let after = {
        let c = b.state.lock().unwrap();
        assert!(c.corner.attempted);
        assert_eq!(c.corner.grant.as_ref().map(|g| g.quote.k), expected);
        let (spent, bytes) = expected.map_or((SETUP_WORK, 0), |k| quote_for(&f, k));
        assert_eq!(c.work, old_work + spent);
        assert_eq!(c.child.work_units, old_child_work + spent);
        assert_eq!(c.child.cache_bytes, old_cache_bytes + bytes);
        let total_work = c.work + c.reserved + c.component_reserved;
        let total_bytes = base
            + c.shared_bytes
            + c.direction.cache_bytes
            + c.child.cache_bytes
            + c.component_rows.admitted_bytes;
        assert!(total_work <= MAX_WORK && total_bytes <= MAX_BYTES);
        if excess == 0 {
            assert_eq!(
                if work_edge { total_work } else { total_bytes },
                if work_edge { MAX_WORK } else { MAX_BYTES }
            );
        }
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    assert!(prepay(&cuts, &f.plan(), b).is_none());
    let c = b.state.lock().unwrap();
    assert_eq!((c.work, c.child.work_units, c.child.cache_bytes), after);
    drop(c);
    // Even an all-mode refusal retains BOTH prepaid original Source replays
    // and ordinary cache bind. No unreserved child-call work is assumed here.
    let cache = completed(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    let root = bound.root_lower().unwrap();
    if expected.is_none() || expected == Some(1) {
        assert!(root < 0.);
    } else {
        assert!(root > 0.13);
    }
    assert_eq!(cuts.proposal_lp_calls, 1);
}

#[test]
fn corner_paid_work_exact_one_unit_fallback_and_all_mode_refusal() {
    for k in [7, 1] {
        for excess in [0, 1] {
            admission_edge(true, k, excess);
        }
    }
}

#[test]
fn corner_paid_bytes_exact_one_byte_fallback_and_all_mode_refusal() {
    for k in [7, 1] {
        for excess in [0, 1] {
            admission_edge(false, k, excess);
        }
    }
}

#[test]
fn corner_unaffordable_setup_is_once_only_and_preserves_old_paid_credits() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, false);
    let b = &cuts.signed.as_ref().unwrap().budget;
    let before_holds = holds(b);
    let before = {
        let mut c = b.state.lock().unwrap();
        c.work = MAX_WORK - c.reserved - c.component_reserved - SETUP_WORK + 1;
        (c.work, c.child.work_units, c.child.cache_bytes)
    };
    for _ in 0..2 {
        assert!(prepay(&cuts, &f.plan(), b).is_none());
    }
    let c = b.state.lock().unwrap();
    assert!(c.corner.attempted && c.corner.grant.is_none());
    assert_eq!((c.work, c.child.work_units, c.child.cache_bytes), before);
    drop(c);
    assert_eq!(holds(b), before_holds);
    let cache = completed(&cuts);
    assert!(
        cache
            .bind(&cuts, end(), &mut || true)
            .unwrap()
            .root_lower()
            .unwrap()
            < -1.88
    );
}

#[test]
fn corner_identical_witness_words_from_another_owner_cannot_reuse_axes() {
    let f = CrossFixture::new();
    let p = f.source();
    let mut cuts = genuine_owner(&p, &f, true);
    {
        let row = cuts.signed.as_ref().unwrap();
        row.replay(&cuts, end(), &mut || true).unwrap();
        assert!(row
            .budget
            .state
            .lock()
            .unwrap()
            .corner
            .grant
            .as_ref()
            .unwrap()
            .axes
            .is_some());
    }
    // Equal contents preserve the old Source/Plan equality check. A fresh Arc
    // deliberately has no first-pass owner identity for the optional product.
    let row = cuts.signed.as_mut().unwrap();
    row.witness = Arc::new(row.witness.as_ref().clone());
    let row = cuts.signed.as_ref().unwrap();
    let second = row.replay_with_cache(&cuts, end(), &mut || true).unwrap();
    assert_eq!(row.stats().source_replays, 2);
    let cache = second.cache.unwrap();
    assert!(
        cache
            .bind(&cuts, end(), &mut || true)
            .unwrap()
            .root_lower()
            .unwrap()
            < -1.88
    );
    let c = row.budget.state.lock().unwrap();
    let grant = c.corner.grant.as_ref().unwrap();
    assert_eq!(grant.stage, 3);
    assert!(grant.axes.is_none());
}

#[test]
fn corner_one_bind_and_sixteen_prices_then_retain_ordinary_fallback() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let cache = completed(&cuts);
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    assert!(bound.root_lower().unwrap() > 0.13);
    for _ in 0..16 {
        assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() > 0.13);
    }
    assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() < -1.88);
    assert!(
        bound.root_lower().unwrap() > 0.13,
        "stored root price is not repriced"
    );
    let b = &cuts.signed.as_ref().unwrap().budget;
    let c = b.state.lock().unwrap();
    let grant = c.corner.grant.as_ref().unwrap();
    assert!(!grant.bind);
    assert_eq!(grant.prices, 0);
    assert_eq!((c.child.attempted_calls, c.child.completed_calls), (17, 17));
    drop(c);
    let second_bind = cache.bind(&cuts, end(), &mut || true).unwrap();
    assert!(second_bind.root_lower().unwrap() < -1.88);
    assert_eq!(cuts.proposal_lp_calls, 1);
}

#[test]
fn corner_sidecar_seal_tamper_burns_only_optional_bind() {
    for corrupt in 0..3 {
        let f = CrossFixture::new();
        let p = f.source();
        let cuts = genuine_owner(&p, &f, true);
        let mut cache = completed(&cuts);
        let saved = {
            let extension = child::corner_extension_for_test(Arc::get_mut(&mut cache).unwrap())
                .as_mut()
                .unwrap();
            let saved = (extension.seal[0], extension.owner, extension.root);
            match corrupt {
                0 => extension.seal[0] ^= 1,
                1 => extension.owner ^= 8,
                _ => extension.root = f64::NAN,
            }
            saved
        };
        let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
        assert!(bound.root_lower().unwrap() < -1.88);
        assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() < -1.88);
        {
            let extension = child::corner_extension_for_test(Arc::get_mut(&mut cache).unwrap())
                .as_mut()
                .unwrap();
            (extension.seal[0], extension.owner, extension.root) = saved;
        }
        // Restoring bytes cannot reopen the already consumed optional bind.
        assert!(
            cache
                .bind(&cuts, end(), &mut || true)
                .unwrap()
                .root_lower()
                .unwrap()
                < -1.88
        );
        assert_eq!(cuts.signed.as_ref().unwrap().stats().source_replays, 2);
    }
}

#[test]
fn corner_valid_sidecar_from_another_cache_cannot_cross_bind() {
    let f = CrossFixture::new();
    let p = f.source();
    let first = genuine_owner(&p, &f, true);
    let second = genuine_owner(&p, &f, true);
    let mut a = completed(&first);
    let mut b = completed(&second);
    std::mem::swap(
        child::corner_extension_for_test(Arc::get_mut(&mut a).unwrap()),
        child::corner_extension_for_test(Arc::get_mut(&mut b).unwrap()),
    );
    // Both complete products have authentic Source/Plan words. Their sealed
    // immutable Cache owners differ, so neither optional capability transfers.
    assert!(
        a.bind(&first, end(), &mut || true)
            .unwrap()
            .root_lower()
            .unwrap()
            < -1.88
    );
    assert!(
        b.bind(&second, end(), &mut || true)
            .unwrap()
            .root_lower()
            .unwrap()
            < -1.88
    );
}

#[test]
fn corner_optional_numeric_refusal_preserves_completed_ordinary_cache() {
    let f = CrossFixture::new();
    let p = f.source();
    let cuts = genuine_owner(&p, &f, true);
    let mut cache = completed(&cuts);
    let extension = child::corner_extension_for_test(Arc::get_mut(&mut cache).unwrap())
        .take()
        .unwrap();
    let Extension { data, .. } = *extension;
    let data = data.expect("fixed corner fixture retains conditional Data");
    let b = &cuts.signed.as_ref().unwrap().budget;
    // Explicit helper fault injection, not another production Source replay:
    // malformed optional coefficients must decline before numeric root pricing.
    let refused = Extension::finish(
        data,
        &[f64::NAN; CrossFixture::D],
        b,
        Arc::as_ptr(&cache) as usize,
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(refused.is_none());
    *child::corner_extension_for_test(Arc::get_mut(&mut cache).unwrap()) = refused;
    let bound = cache.bind(&cuts, end(), &mut || true).unwrap();
    assert!(bound.root_lower().unwrap() < -1.88);
    assert!(bound.lower_zero(&[], end(), &mut || true).unwrap() < -1.88);
    assert_eq!(cuts.signed.as_ref().unwrap().stats().source_replays, 2);
}

#[path = "single_hidden_hull_mip_signed_corner_base_call_tests.rs"]
mod base_call_tests;
