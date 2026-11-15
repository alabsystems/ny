// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Owner controls use genuine source/descriptor arithmetic, not returned scalars.
use super::*;
use crate::layers::{FlattenLayer, LinearLayer, ReLULayer, ReshapeLayer};
use crate::network::raw_root_cut::raw_two_pair_root_cut_test_support::{
    baseline, fixture, JointBackend,
};
use crate::GraphNode;
use ndarray::{arr1, arr2, Array2};
use ny_core::{
    GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment,
    ResidentLowerCutCarrier, ResidentTwoPairLowerCutCarrier,
};

struct Route {
    support: bool,
    single: bool,
    joint: bool,
}
impl GemmEngine for Route {
    fn gemm_f32(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        panic!("routing cannot dispatch")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for Route {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        panic!("routing cannot dispatch")
    }
    fn provides_bounded_support4(&self) -> bool {
        self.support
    }
    fn provides_bounded_lower_cut(&self) -> bool {
        self.single
    }
    fn provides_bounded_two_pair_lower_cut(&self) -> bool {
        self.joint
    }
}
fn same(a: &dyn GemmEngine, b: &dyn GemmEngine) -> bool {
    std::ptr::eq(
        std::ptr::from_ref::<dyn GemmEngine>(a).cast::<()>(),
        std::ptr::from_ref::<dyn GemmEngine>(b).cast::<()>(),
    )
}

#[test]
fn joint_ready_route_keeps_cpu_custody_and_does_not_hide_new_global_owner() {
    let old = Route {
        support: true,
        single: true,
        joint: false,
    };
    let new = Route {
        support: true,
        single: false,
        joint: true,
    };
    let both = Route {
        support: true,
        single: true,
        joint: true,
    };
    let incomplete = Route {
        support: false,
        single: true,
        joint: true,
    };
    let r =
        select_ready_methods(Some(&old), false, || [Some(&incomplete), Some(&new), None]).unwrap();
    assert!(same(r.single.unwrap(), &old));
    assert!(same(r.joint.unwrap(), &new));
    let r = select_ready_methods(Some(&both), false, || {
        [None, None, None] // A ready Triplet owner may now exist independently.
    })
    .unwrap();
    assert!(same(r.single.unwrap(), &both));
    assert!(same(r.joint.unwrap(), &both));
    assert!(select_ready_methods(Some(&both), true, || panic!(
        "CPU-only forbids every lookup"
    ))
    .is_none());
    assert!(select_ready_methods(None, false, || [Some(&incomplete), None, None]).is_none());
    let r = select_ready_methods(Some(&old), false, || [None, None, None]).unwrap();
    assert!(r.joint.is_none());
    assert!(same(r.single.unwrap(), &old));
    assert!(raw_load_opportunity(Some(&new), false, false));
    assert!(!raw_load_opportunity(Some(&new), true, true));
}

fn gap_fixture() -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    let mut w = Array2::<f32>::zeros((6, 6));
    for (i, s) in [16., 8., 4., 2., 0.25, 0.125].into_iter().enumerate() {
        w[[i, i]] = s;
    }
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "reshape",
        Layer::Reshape(ReshapeLayer::new(vec![1, 6])),
        vec!["relu".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::new(1)),
        vec!["reshape".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[-1., -2., -4., -8., -16., -32.]]), None).unwrap()),
        vec!["flat".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 6]).into_dyn(), arr1(&[1.; 6]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
fn ledger(end: Instant, work: u64) -> WorkBudget {
    WorkBudget::new(Some(end), QUOTE_SCRATCH, 0, work, 1, Duration::ZERO).unwrap()
}
fn setup(b: &WorkBudget) -> WorkLease {
    b.lease(WorkRequest {
        scratch_bytes: QUOTE_SCRATCH,
        proposals: 1,
        ..WorkRequest::default()
    })
    .unwrap()
}

#[test]
fn joint_gap_ranking_transports_flatten_reshape_and_freezes_pair_order() {
    let (g, x, b) = gap_fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let work = JOINT_SELECTION_FIXED_WORK + JOINT_SELECTION_WORK_PER_CELL * 6 * 2;
    let budget = ledger(end, work);
    let mut lease = setup(&budget);
    let (p, s) = joint_target(&g, &x, &b, &mut lease, &[1.], "relu", end).unwrap();
    // Scores are [8,8,8,8,2,2], while magnitude-only picks 5,4,3,2.
    // Equal scores resolve by original coordinate; indices freeze before seeds.
    assert_eq!(p, [[0, 1], [2, 3]]);
    assert_eq!(s, [0.5, 2.]);
    assert_eq!(budget.receipt().unwrap().work_reserved, work);
}

#[test]
fn joint_selection_reserves_fixed_and_full_scan_before_refusal() {
    let (g, x, b) = gap_fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let total = JOINT_SELECTION_FIXED_WORK + JOINT_SELECTION_WORK_PER_CELL * 6 * 2;
    for (cap, spent) in [
        (JOINT_SELECTION_FIXED_WORK - 1, 0),
        (total - 1, JOINT_SELECTION_FIXED_WORK),
    ] {
        let budget = ledger(end, cap);
        let mut lease = setup(&budget);
        assert!(joint_target(&g, &x, &b, &mut lease, &[1.], "relu", end).is_none());
        assert_eq!(budget.receipt().unwrap().work_reserved, spent);
    }
    let budget = ledger(end, total);
    let mut lease = setup(&budget);
    assert!(joint_target(
        &g,
        &x,
        &b,
        &mut lease,
        &[1.],
        "relu",
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
    )
    .is_none());
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        JOINT_SELECTION_FIXED_WORK
    );
    // Includes both opaque proposal/completion unions and bounded ranking data.
    assert!(
        size_of::<RootCutObservation>()
            + size_of::<RootProposal<'_, '_>>()
            + size_of::<RootCompletion>()
            + size_of::<[(usize, f64, f64); 4]>()
            + 8192
            < QUOTE_SCRATCH
    );
}

#[derive(Default)]
struct BothBackend {
    joint: JointBackend,
}
impl GemmEngine for BothBackend {
    fn gemm_f32(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        panic!("checked consumer cannot widen ordinary permission")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for BothBackend {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        panic!("checked consumer cannot widen ordinary permission")
    }
    fn provides_bounded_support4(&self) -> bool {
        true
    }
    fn provides_bounded_lower_cut(&self) -> bool {
        true
    }
    fn provides_bounded_two_pair_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_support4_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        end: Instant,
    ) -> ny_core::Result<GpuCrownResult> {
        self.joint
            .crown_backward_gpu_resnet_support4_with_deadline(s, q, l, u, end)
    }
    fn crown_backward_gpu_resnet_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentLowerCutCarrier,
        end: Instant,
    ) -> ny_core::Result<f32> {
        self.joint
            .ordinary
            .crown_backward_gpu_resnet_lower_cut_with_deadline(s, q, l, u, c, end)
    }
    fn crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTwoPairLowerCutCarrier,
        end: Instant,
    ) -> ny_core::Result<f32> {
        self.joint
            .crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(s, q, l, u, c, end)
    }
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

#[test]
fn joint_genuine_root_consumer_preserves_upper_negative_truth_and_once_only_owner() {
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let c = config(&owner);
        let engine = BothBackend::default();
        let mut bounds = [old];
        {
            let _cpu = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
            assert!(try_tighten(
                &c,
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut bounds,
                &b,
                None,
                Some(&engine),
                Some(end)
            )
            .is_none());
            assert!(!owner.attempted.load(Ordering::Acquire));
            assert_eq!(bounds, [old]);
        }
        assert_eq!(
            try_tighten(
                &c,
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut bounds,
                &b,
                None,
                Some(&engine),
                Some(end)
            ),
            Some(0)
        );
        assert_eq!(bounds[0].0, 0.25 + shift);
        assert_eq!(bounds[0].1.to_bits(), old.1.to_bits());
        assert_eq!(
            engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
            1
        );
        assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.joint.ordinary.cut_calls.load(Ordering::SeqCst), 0);
        let mut again = [old];
        assert!(try_tighten(
            &c,
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
        assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn joint_preexecution_quote_refusal_pays_same_owner_single_fallback() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let planner = QueryOwner::new(Some(end), true).unwrap();
    let mut plan = planner
        .budget
        .lease(WorkRequest {
            scratch_bytes: QUOTE_SCRATCH,
            work: QUOTE_WORK,
            proposals: 1,
            ..WorkRequest::default()
        })
        .unwrap();
    let (name, pair, scale) = target(&g, &x, &b, &mut plan, &[1.], end).unwrap();
    let (pairs, scales) = joint_target(&g, &x, &b, &mut plan, &[1.], name, end).unwrap();
    let paid_setup = planner.budget.receipt().unwrap().work_reserved;
    let limits = RawRootCutLimits {
        max_nodes: MAX_NODES,
        max_scratch_bytes: 32 << 20,
        max_work: MAX_WORK,
    };
    let single = quote_raw_root_cut(
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
    let single_work = single.work();
    assert_eq!(single.candidate_count(), 2);
    let remaining = QUOTE_WORK + single_work;
    assert!(quote_raw_two_pair_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        name,
        pairs,
        scales,
        RawRootCutLimits {
            max_work: remaining,
            ..limits
        },
        end,
        &mut || true
    )
    .is_none());
    drop(plan);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let precharge = MAX_WORK - paid_setup - remaining;
    drop(
        owner
            .budget
            .lease(WorkRequest {
                work: precharge,
                ..WorkRequest::default()
            })
            .unwrap(),
    );
    let c = config(&owner);
    let engine = BothBackend::default();
    let mut bounds = [old];
    assert_eq!(
        try_tighten(
            &c,
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut bounds,
            &b,
            None,
            Some(&engine),
            Some(end)
        ),
        Some(0)
    );
    assert!(bounds[0].0 > old.0 && bounds[0].0 < 0.);
    assert_eq!(bounds[0].1.to_bits(), old.1.to_bits());
    assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
        1
    );
    assert_eq!(engine.joint.ordinary.cut_calls.load(Ordering::SeqCst), 2);
    assert_eq!(owner.budget.receipt().unwrap().work_reserved, MAX_WORK);
    assert_eq!(owner.budget.receipt().unwrap().proposals_reserved, 1);
}

#[test]
fn joint_entered_refusal_keeps_original_scalar_without_single_retry() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let c = config(&owner);
    let engine = BothBackend::default();
    engine.joint.refuse_joint.store(1, Ordering::SeqCst);
    let mut bounds = [old];
    assert!(try_tighten(
        &c,
        &g,
        &x,
        &[vec![1.]],
        &[0.],
        &mut bounds,
        &b,
        None,
        Some(&engine),
        Some(end)
    )
    .is_none());
    assert_eq!(bounds, [old]);
    assert_eq!(
        engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
        1
    );
    assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.joint.ordinary.cut_calls.load(Ordering::SeqCst), 0);
    assert!(owner.attempted.load(Ordering::Acquire));
    assert!(try_tighten(
        &c,
        &g,
        &x,
        &[vec![1.]],
        &[0.],
        &mut bounds,
        &b,
        None,
        Some(&engine),
        Some(end)
    )
    .is_none());
    assert_eq!(engine.joint.joint_calls.load(Ordering::SeqCst), 1);
    assert_eq!(bounds, [old]);
}
