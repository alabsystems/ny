// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::layers::{BatchNormLayer, LinearLayer, ReLULayer};
use crate::network::raw_root_cut::raw_triplet_root_cut_test_support::{
    cycle_fixture, TripletBackend,
};
use crate::network::raw_root_cut::raw_two_pair_root_cut_test_support::{baseline, JointBackend};
use crate::network::GraphNode;
use ndarray::{arr1, arr2};

fn setup(end: Instant, work: u64) -> (WorkBudget, WorkLease) {
    let b = WorkBudget::new(
        Some(end),
        QUOTE_SCRATCH + FAMILY_SCRATCH,
        0,
        work,
        1,
        Duration::ZERO,
    )
    .unwrap();
    let l = b
        .lease(WorkRequest {
            scratch_bytes: QUOTE_SCRATCH + FAMILY_SCRATCH,
            proposals: 1,
            ..WorkRequest::default()
        })
        .unwrap();
    (b, l)
}
fn config(owner: &Arc<QueryOwner>) -> BetaCrownConfig {
    let defaults = BetaCrownConfig::default();
    BetaCrownConfig {
        certified_root_cut_owner: Some(Arc::clone(owner)),
        certified_root_cut_raw_source: true,
        verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
        ..defaults
    }
}
fn permuted_diamonds(
    raw_bn: bool,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[
                    [0.5, 0.5, 0., 0.],
                    [0., 0., 0.5, 0.5],
                    [0.5, -0.5, 0., 0.],
                    [0., 0., 0.5, -0.5],
                    [1., 0., 1., 0.],
                ]),
                Some(arr1(&[0., 0., 0., 0., 4.])),
            )
            .unwrap(),
        ),
    ));
    let parent = if raw_bn {
        let bn = BatchNormLayer::new(
            &arr1(&[1.; 5]).into_dyn(),
            &arr1(&[0.; 5]).into_dyn(),
            &arr1(&[0.; 5]).into_dyn(),
            &arr1(&[1.; 5]).into_dyn(),
            0.,
        )
        .unwrap();
        g.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(bn),
            vec!["pre".into()],
        ));
        "bn"
    } else {
        "pre"
    };
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec![parent.into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[-2., -2., -2., -2., 1.]]), Some(arr1(&[-1.75]))).unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 4]).into_dyn(), arr1(&[1.; 4]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
#[test]
fn family_ready_routes_new_global_without_cpu_override_or_lazy_dispatch() {
    let old = JointBackend::default();
    let new = TripletBackend::default();
    let r = select_ready_methods(Some(&old), false, || [Some(&new), None, None]).unwrap();
    assert!(std::ptr::eq(
        std::ptr::from_ref(r.joint.unwrap()).cast::<()>(),
        (&raw const old).cast::<()>()
    ));
    assert!(std::ptr::eq(
        std::ptr::from_ref(r.triplet.unwrap()).cast::<()>(),
        (&raw const new).cast::<()>()
    ));
    assert!(
        select_ready_methods(Some(&new), true, || panic!("CPU guard precedes getter")).is_none()
    );
    let r = select_ready_methods(
        Some(&new),
        false,
        || [None, None, None], // Local-L1 may be available in a ready slot.
    )
    .unwrap();
    assert!(r.single.is_some() && r.joint.is_some() && r.triplet.is_some());
    assert!(raw_load_opportunity(Some(&new), false, false));
    assert_eq!(new.triplet_calls.load(Ordering::SeqCst), 0);
    assert_eq!(new.joint.joint_calls.load(Ordering::SeqCst), 0);
}
#[test]
fn family_screen_selects_cycle_and_nonadjacent_diamonds_with_raw_bn_advice() {
    let end = Instant::now() + Duration::from_secs(30);
    let (g, x, b) = cycle_fixture(0.);
    let (_, mut lease) = setup(end, MAX_WORK);
    assert_eq!(
        family_target(&g, &x, &b, &mut lease, &[1.], "relu", end, true),
        Some(FamilyChoice::Triplet {
            neurons: [0, 1, 2],
            weights: [1.; 3],
            multiplier: 1.
        })
    );
    for bn in [false, true] {
        let (g, x, b) = permuted_diamonds(bn);
        let (_, mut lease) = setup(end, MAX_WORK);
        assert_eq!(
            family_target(&g, &x, &b, &mut lease, &[1.], "relu", end, true),
            Some(FamilyChoice::Joint {
                pairs: [[0, 2], [1, 3]],
                scales: [1.; 2]
            })
        );
        let (_, mut lease) = setup(end, MAX_WORK);
        assert!(matches!(
            family_target(&g, &x, &b, &mut lease, &[1.], "relu", end, false),
            Some(FamilyChoice::Triplet { .. })
        ));
    }
}
#[test]
fn family_screen_fixed_scan_and_scratch_limits_are_prepaid_and_monotone() {
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let rank = JOINT_SELECTION_FIXED_WORK + JOINT_SELECTION_WORK_PER_CELL * 3 * 2;
    let local = (1 << 24) + 8192;
    let scan = 1024 * 3;
    let total = rank + local + scan;
    for (cap, spent, ok) in [
        (rank + local - 1, rank, false),
        (total - 1, rank + local, false),
        (total, total, true),
    ] {
        let (budget, mut l) = setup(end, cap);
        assert_eq!(
            family_target(&g, &x, &b, &mut l, &[1.], "relu", end, true).is_some(),
            ok
        );
        assert_eq!(budget.receipt().unwrap().work_reserved, spent);
        assert_eq!(
            budget.receipt().unwrap().live_scratch_bytes,
            QUOTE_SCRATCH + FAMILY_SCRATCH
        );
    }
    let (budget, mut l) = setup(end, total);
    assert!(family_target(
        &g,
        &x,
        &b,
        &mut l,
        &[1.],
        "relu",
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        true
    )
    .is_none());
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        JOINT_SELECTION_FIXED_WORK
    );
    // Both live claims must fit; the execution cap subtracts this same extra8KiB.
    let b = WorkBudget::new(
        Some(end),
        QUOTE_SCRATCH + FAMILY_SCRATCH + 123,
        0,
        MAX_WORK,
        1,
        Duration::ZERO,
    )
    .unwrap();
    let _s = b
        .lease(WorkRequest {
            scratch_bytes: QUOTE_SCRATCH + FAMILY_SCRATCH,
            ..WorkRequest::default()
        })
        .unwrap();
    assert!(b
        .lease(WorkRequest {
            scratch_bytes: 124,
            ..WorkRequest::default()
        })
        .is_err());
    assert!(b
        .lease(WorkRequest {
            scratch_bytes: 123,
            ..WorkRequest::default()
        })
        .is_ok());
}
#[test]
fn family_genuine_root_cycle_and_diamonds_keep_unsafe_truth_upper_and_once_only_custody() {
    for shift in [0., -0.5] {
        let (g, x, b) = cycle_fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let engine = TripletBackend::default();
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
        assert_eq!(out[0], (0.25 + shift, old.1));
        assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 0);
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
    }
    let (g, x, b) = permuted_diamonds(false);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let engine = TripletBackend::default();
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
    assert_eq!(out[0], (0.25, old.1));
    assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 1);
}
#[test]
fn family_entered_triplet_refusal_never_retries_old_numerical_method() {
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let engine = TripletBackend::default();
    engine.refuse_triplet.store(1, Ordering::SeqCst);
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
        Some(end)
    )
    .is_none());
    assert_eq!(out, [old]);
    assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
        1
    );
    assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.joint.ordinary.cut_calls.load(Ordering::SeqCst), 0);
}
#[test]
fn family_preexecution_quote_refusal_pays_screen_and_old_fallback_from_same_ledger() {
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let (planner, mut setup) = setup(end, MAX_WORK);
    setup.reserve_more_work(QUOTE_WORK).unwrap();
    let (name, pair, scale) = target(&g, &x, &b, &mut setup, &[1.], end).unwrap();
    assert!(family_target(&g, &x, &b, &mut setup, &[1.], name, end, true).is_some());
    let paid = planner.receipt().unwrap().work_reserved;
    let limits = RawRootCutLimits {
        max_nodes: MAX_NODES,
        max_work: MAX_WORK,
        max_scratch_bytes: 32 << 20,
    };
    let q = quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        name,
        pair,
        scale,
        limits,
        end,
        &mut || true,
    )
    .unwrap();
    let rank = JOINT_SELECTION_FIXED_WORK + JOINT_SELECTION_WORK_PER_CELL * 3 * 2;
    let rest = 2 * QUOTE_WORK + rank + q.work();
    assert!(quote_raw_triplet_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        name,
        [0, 1, 2],
        [1.; 3],
        1.,
        RawRootCutLimits {
            max_work: rest,
            ..limits
        },
        end,
        &mut || true
    )
    .is_none());
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let pre = owner
        .budget
        .lease(WorkRequest {
            work: MAX_WORK - paid - rest,
            ..WorkRequest::default()
        })
        .unwrap();
    drop(pre);
    let engine = TripletBackend::default();
    let mut out = [old];
    let _ = try_tighten(
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
    );
    assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.joint.ordinary.cut_calls.load(Ordering::SeqCst), 2);
    assert_eq!(owner.budget.receipt().unwrap().work_reserved, MAX_WORK);
    assert_eq!(out[0].1.to_bits(), old.1.to_bits());
}
