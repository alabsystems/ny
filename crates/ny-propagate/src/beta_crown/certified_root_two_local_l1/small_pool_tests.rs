// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent scales recover a relation the common Q8 scale cannot represent.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, Array2};

fn six_cycle_fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut w = Array2::<f32>::zeros((6, 6));
    for j in [0, 3] {
        w[[j, j]] = 0.5;
        w[[j, j + 1]] = 0.5;
        w[[j + 1, j + 1]] = 0.5;
        w[[j + 1, j + 2]] = 0.5;
        w[[j + 2, j + 2]] = 0.5;
        w[[j + 2, j]] = -0.5;
    }
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 6), vec![-1024., -1024., -1024., -1., -1., -1.])
                    .unwrap(),
                Some(arr1(&[2050.25 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let input = BoundedTensor::new(arr1(&[-1.; 6]).into_dyn(), arr1(&[1.; 6]).into_dyn()).unwrap();
    let bounds = g.collect_node_bounds(&input).unwrap();
    (g, input, bounds)
}

#[test]
fn small_pool_original_owner_separates_top_triplet_and_preserves_unsafe_shift() {
    // Exact integer oracle. Concavity of b-sum(a_i ReLU(z_i)), a_i>0,
    // puts a minimum at a cube vertex. z2 is twice each original preactivation.
    let mut minimum4 = i32::MAX;
    let mut support2 = [0; 2];
    for mask in 0usize..64 {
        let x: [i32; 6] = std::array::from_fn(|j| if mask & (1 << j) == 0 { -1 } else { 1 });
        let z2: [i32; 6] = std::array::from_fn(|j| {
            let k = j / 3 * 3;
            match j % 3 {
                0 => x[k] + x[k + 1],
                1 => x[k + 1] + x[k + 2],
                _ => x[k + 2] - x[k],
            }
        });
        for g in 0..2 {
            support2[g] = support2[g].max(z2[3 * g..3 * g + 3].iter().map(|z| z.abs()).sum());
        }
        let value4 = 8201
            - 2048 * z2[..3].iter().map(|z| (*z).max(0)).sum::<i32>()
            - 2 * z2[3..].iter().map(|z| (*z).max(0)).sum::<i32>();
        minimum4 = minimum4.min(value4);
    }
    assert_eq!(support2, [4, 4]);
    assert_eq!(minimum4, 1);
    for shift in [0., -0.5] {
        let (g, x, b) = six_cycle_fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= -512.25 + shift && old.0 > -512.3 + shift);
        let (_, mut lease) = setup(end, MAX_WORK);
        // The refusal is caused by the selected small coefficient rounding to
        // zero, not by absence of negative crossings or a source mismatch.
        assert_eq!((256f64 / 1024.).round(), 0.);
        assert!(local_target(&g, &x, &b, &mut lease, &[1.], "relu", end).is_none());
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
                    Some(end),
                ),
                Some(0)
            );
            assert_eq!(out[0].1.to_bits(), old.1.to_bits());
            if enabled {
                assert!(out[0].0 > 0.249 + shift && out[0].0 <= 0.25 + shift);
                // A strict box proof skips the prefix; the unsafe shift must
                // complete its admitted prefix and cannot publish a proof.
                assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
                assert_eq!(engine.old_calls(), 0);
            } else {
                // Capability-masked control executes the previous top-triplet
                // route, with a genuine descriptor backend and fresh support.
                assert!(out[0].0 <= -0.25 + shift && out[0].0 > -0.251 + shift);
                assert_eq!(count(&engine.old.triplet_calls), 1);
                assert_eq!(count(&engine.bundle.bundle_calls), 0);
                assert_eq!(count(&engine.local.local_calls), 0);
            }
            assert!(out[0].0 <= minimum4 as f32 / 4. + shift);
            assert!(owner.attempted.load(Ordering::Acquire));
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
                Some(end),
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
fn small_pool_common_scale_routes_and_exact_selection_debits() {
    let (mut g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let scan = (1 << 24) + 8192 + 1024 * 17 * 2;
    for count in 3..=8 {
        let mut values = vec![1.; 17];
        values[..count].fill(-2.);
        output_weights(&mut g, &values);
        let (budget, mut lease) = setup(end, scan);
        let mut obs = AdmissionObservation::default();
        assert!(bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).is_none());
        assert!(obs.scan_complete);
        assert_eq!(obs.negative_crossing, count);
        assert_eq!(
            obs.reason,
            if count < 6 {
                "insufficient-negative-crossings"
            } else {
                "single-local-representable"
            }
        );
        assert_eq!(budget.receipt().unwrap().work_reserved, scan);
    }
    for count in 6..=8 {
        let mut values = vec![1.; 17];
        values[..count - 3].fill(-1024.);
        values[count - 3..count].fill(-1.);
        output_weights(&mut g, &values);
        let (budget, mut lease) = setup(end, scan);
        let mut obs = AdmissionObservation::default();
        let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).unwrap();
        assert_eq!(choice.active, [count - 3, 3]);
        assert_eq!(choice.multipliers, [512., 0.5]);
        assert_eq!(obs.groups, choice.active);
        assert!(obs.scan_complete && obs.failed_group.is_none());
        for group in 0..2 {
            assert_eq!(
                &choice.weights[group][..choice.active[group]],
                vec![1.; choice.active[group]]
            );
            assert!(choice.weights[group][choice.active[group]..]
                .iter()
                .all(|v| v.to_bits() == 0));
        }
        assert_eq!(budget.receipt().unwrap().work_reserved, scan);
    }
}

#[test]
fn small_pool_four_plus_four_rescue_has_paid_work_and_keeps_all_coordinates() {
    let (mut g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let mut values = vec![1.; 17];
    values[..4].fill(-1024.);
    values[4..8].fill(-1.);
    output_weights(&mut g, &values);
    let scan = (1 << 24) + 8192 + 1024 * 17 * 2;
    let work = scan + two_local_l1::WINDOW_RESCUE_WORK;
    for cap in [work - 1, work] {
        let (budget, mut lease) = setup(end, cap);
        let mut obs = AdmissionObservation::default();
        let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs);
        assert!(obs.scan_complete);
        if cap < work {
            assert!(choice.is_none());
            assert_eq!((obs.stage, obs.reason), ("window-rescue", "window-work"));
            assert_eq!(budget.receipt().unwrap().work_reserved, scan);
        } else {
            let choice = choice.unwrap();
            assert_eq!(choice.active, [4, 4]);
            assert_eq!(
                choice.neurons,
                [[0, 1, 2, 3, 0, 0, 0, 0], [4, 5, 6, 7, 0, 0, 0, 0]]
            );
            assert_eq!(choice.multipliers, [512., 0.5]);
            assert_eq!(budget.receipt().unwrap().work_reserved, work);
        }
    }
}

#[test]
fn small_pool_rescue_refuses_incompatible_scales_cancellation_and_deadline() {
    let end = Instant::now() + Duration::from_secs(30);
    let work = two_local_l1::WINDOW_RESCUE_WORK;
    for count in 6..=8 {
        // Each separated cluster has at most two coordinates. A live rescue
        // can now retain two such groups; cancellation/deadline still refuse.
        let mut ranked = [(0usize, 1f64, 0f64); 16];
        for (i, item) in ranked[..count].iter_mut().enumerate() {
            *item = (i, 1., 1024f64.powi((i / 2) as i32));
        }
        for state in 0..3 {
            let memory = MemoryAuthority::new(0);
            let cancel = CancellationToken::default();
            let budget = WorkBudget::new_in(
                &memory,
                &cancel,
                WorkLimits {
                    deadline: Some(end),
                    scratch_bytes: 0,
                    persistent_bytes: 0,
                    max_work: work,
                    max_proposals: 1,
                    certificate_reserve: Duration::ZERO,
                    reserve: WorkReserve::default(),
                },
            )
            .unwrap();
            let mut lease = budget.lease(WorkRequest::default()).unwrap();
            if state == 1 {
                cancel.cancel();
            }
            let deadline = if state == 2 {
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap()
            } else {
                end
            };
            let mut obs = AdmissionObservation::default();
            let choice =
                two_local_l1::rescue_windows(&ranked, count, &mut lease, deadline, &mut obs);
            if state == 0 {
                let choice = choice.unwrap();
                assert_eq!(choice.active.iter().sum::<usize>(), 4);
                assert!(choice.active.iter().all(|n| *n == 2));
            } else {
                assert!(choice.is_none());
            }
            assert_eq!(
                budget.receipt().unwrap().work_reserved,
                if state == 1 { 0 } else { work }
            );
        }
    }
    assert!(two_local_l1::rescue_storage_bytes().unwrap() <= FAMILY_SCRATCH);
}

#[test]
fn small_pool_required_prefix_failure_cannot_publish_box_candidate() {
    let (g, x, b) = six_cycle_fixture(-0.5);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let engine = WithFallback::new(true);
    engine.bundle.refuse_bundle.store(1, Ordering::SeqCst);
    let mut out = [old];
    assert!(try_tighten(
        &config(&owner),
        &g,
        &x,
        &[vec![1.]],
        &[0.],
        &mut out,
        &b,
        None,
        Some(&engine),
        Some(end),
    )
    .is_none());
    assert_eq!(out, [old]);
    assert_eq!(count(&engine.bundle.bundle_calls), 1);
    assert_eq!(engine.old_calls(), 0);
    assert!(owner.attempted.load(Ordering::Acquire));
}

#[test]
fn small_head_advice_default_owner_uses_interior_joint_scale_and_keeps_unsafe_truth() {
    use crate::network::raw_root_cut::raw_two_local_l1_root_cut_test_support::small_advice_fixture;
    for shift in [0., -256.] {
        let (g, x, b) = small_advice_fixture(shift);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        assert!(
            old.0 <= -64.5 + shift && old.0 > -64.51 + shift,
            "baseline={old:?}, shift={shift}"
        );
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let engine = WithFallback::new(true);
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
        assert!(out[0].0 > 63.9 + shift && out[0].0 <= 64. + shift);
        assert_eq!(out[0].1.to_bits(), old.1.to_bits());
        assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
        assert_eq!(engine.old_calls(), 0);
        assert!(owner.attempted.load(Ordering::Acquire));
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
        assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
    }
}
