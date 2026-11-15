// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::network::raw_root_cut::raw_two_local_l1_root_cut_test_support::low_arity_fixture;

#[test]
fn low_arity_default_owner_closes_original_three_plus_two_and_keeps_unsafe_shift() {
    for shift in [0., -0.5] {
        let (g, x, b) = low_arity_fixture(shift);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        for enabled in [false, true] {
            let owner = QueryOwner::new(Some(end), true).unwrap();
            let engine = WithFallback::new(enabled);
            let mut out = [old];
            assert_eq!(
                try_tighten(
                    &config(&owner),
                    &g,
                    &x,
                    &[vec![1.]],
                    &[0.],
                    &mut out,
                    &b,
                    None,
                    Some(&engine),
                    Some(end)
                ),
                Some(0)
            );
            assert_eq!(out[0].1.to_bits(), old.1.to_bits());
            if enabled {
                assert!(out[0].0 > 0.24 + shift && out[0].0 <= 0.25 + shift);
                assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
                assert_eq!(engine.old_calls(), 0);
            } else {
                assert!(out[0].0 <= -0.25 + shift && out[0].0 > -0.251 + shift);
                assert_eq!(count(&engine.old.triplet_calls), 1);
            }
            let calls = engine.old_calls() + count(&engine.bundle.bundle_calls);
            let mut again = [old];
            assert!(try_tighten(
                &config(&owner),
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut again,
                &b,
                None,
                Some(&engine),
                Some(end)
            )
            .is_none());
            assert_eq!(again, [old]);
            assert_eq!(
                engine.old_calls() + count(&engine.bundle.bundle_calls),
                calls
            );
        }
    }
}

#[test]
fn low_arity_window_rescue_pays_exact_work_retains_full_objective_and_canonical_groups() {
    let (g, x, b) = low_arity_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let scan = (1 << 24) + 8192 + 1024 * 6 * 2;
    let complete = scan + two_local_l1::WINDOW_RESCUE_WORK;
    for cap in [complete - 1, complete] {
        let (budget, mut lease) = setup(end, cap);
        let mut obs = AdmissionObservation::default();
        let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs);
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            if cap < complete { scan } else { complete }
        );
        if cap < complete {
            assert!(choice.is_none());
            assert_eq!(obs.reason, "window-work");
            continue;
        }
        let c = choice.unwrap();
        assert_eq!(c.active, [2, 3]);
        assert_eq!(c.multipliers, [0.5, 512.]);
        assert_eq!(
            c.neurons,
            [[3, 4, 0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0, 0, 0]]
        );
        assert_eq!(obs.negative_crossing, 6);
        assert_eq!(obs.groups, [2, 3]);
        let Layer::Linear(last) = g.node("out").unwrap().layer() else {
            unreachable!()
        };
        assert_eq!(last.in_features(), 6);
        assert_eq!(last.weight[[0, 5]], -1. / 1024.);
    }
    assert!(two_local_l1::rescue_storage_bytes().unwrap() <= FAMILY_SCRATCH);
}
