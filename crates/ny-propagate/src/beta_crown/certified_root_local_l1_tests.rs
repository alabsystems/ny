// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::network::raw_root_cut::raw_local_l1_root_cut_test_support::{
    baseline, fixture, LocalBackend,
};
use crate::network::raw_root_cut::raw_triplet_root_cut_test_support::{
    cycle_fixture, TripletBackend,
};
use crate::network::raw_root_cut::raw_two_pair_root_cut_test_support::fixture as two_diamonds;
use ny_core::{
    GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment,
    ResidentLocalL1LowerCutCarrier, ResidentLowerCutCarrier, ResidentTripletLowerCutCarrier,
    ResidentTwoPairLowerCutCarrier,
};
use std::sync::atomic::AtomicUsize;
#[derive(Default)]
struct Both {
    local: LocalBackend,
    old: TripletBackend,
    single_only: bool,
}
impl GemmEngine for Both {
    fn gemm_f32(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        panic!("no ordinary dispatch")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for Both {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        panic!("no ordinary dispatch")
    }
    fn provides_bounded_support4(&self) -> bool {
        true
    }
    fn provides_bounded_lower_cut(&self) -> bool {
        true
    }
    fn provides_bounded_two_pair_lower_cut(&self) -> bool {
        !self.single_only
    }
    fn provides_bounded_triplet_lower_cut(&self) -> bool {
        !self.single_only
    }
    fn provides_bounded_local_l1_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_support4_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        d: Instant,
    ) -> ny_core::Result<GpuCrownResult> {
        self.old
            .crown_backward_gpu_resnet_support4_with_deadline(s, q, l, u, d)
    }
    fn crown_backward_gpu_resnet_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentLowerCutCarrier,
        d: Instant,
    ) -> ny_core::Result<f32> {
        self.old
            .crown_backward_gpu_resnet_lower_cut_with_deadline(s, q, l, u, c, d)
    }
    fn crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTwoPairLowerCutCarrier,
        d: Instant,
    ) -> ny_core::Result<f32> {
        self.old
            .crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(s, q, l, u, c, d)
    }
    fn crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTripletLowerCutCarrier,
        d: Instant,
    ) -> ny_core::Result<f32> {
        self.old
            .crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(s, q, l, u, c, d)
    }
    fn crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentLocalL1LowerCutCarrier,
        d: Instant,
    ) -> ny_core::Result<f32> {
        self.local
            .crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(s, q, l, u, c, d)
    }
}
fn count(v: &AtomicUsize) -> usize {
    v.load(Ordering::SeqCst)
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
            ..WorkRequest::default()
        })
        .unwrap();
    (b, l)
}
#[test]
fn local_ready_capability_is_independent_of_support4_old_local_and_cpu_custody() {
    let old = TripletBackend::default();
    let local = LocalBackend::default();
    let ready = select_ready_methods(Some(&old), false, || [Some(&local), None, None]).unwrap();
    assert!(std::ptr::eq(
        std::ptr::from_ref(ready.local_l1.unwrap()).cast::<()>(),
        (&raw const local).cast::<()>()
    ));
    assert!(std::ptr::eq(
        std::ptr::from_ref(ready.triplet.unwrap()).cast::<()>(),
        (&raw const old).cast::<()>()
    ));
    let ready = select_ready_methods(Some(&local), false, || [None, None, None]).unwrap();
    assert!(
        ready.local_l1.is_some()
            && ready.single.is_none()
            && ready.joint.is_none()
            && ready.triplet.is_none()
    );
    assert!(!local.provides_bounded_support4());
    assert!(raw_load_opportunity(Some(&local), false, false));
    assert!(!raw_load_opportunity(Some(&local), false, true));
    assert!(select_ready_methods(Some(&local), true, || panic!(
        "CPU guard before ready getters"
    ))
    .is_none());
    assert!(select_ready_methods(None, false, || [None, None, None]).is_none());
    let both = Both::default();
    // Old capabilities do not hide a ready independent bundle backend.
    assert!(select_ready_methods(Some(&both), false, || [None, None, None]).is_some());
    assert_eq!(count(&local.local_calls), 0);
}
#[test]
fn local_top8_prepays_complete_scan_and_retains_canonical_q8_selection() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let fixed = (1 << 24) + 8192;
    let scan = 512 * 9 * 2;
    for (work, ok, spent) in [
        (fixed - 1, false, 0),
        (fixed + scan - 1, false, fixed),
        (fixed + scan, true, fixed + scan),
    ] {
        let (budget, mut lease) = setup(end, work);
        let choice = local_target(&g, &x, &b, &mut lease, &[1.], "relu", end);
        assert_eq!(choice.is_some(), ok);
        assert_eq!(budget.receipt().unwrap().work_reserved, spent);
        if let Some(c) = choice {
            assert_eq!(c.active, 8);
            assert_eq!(c.neurons, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(c.weights, [1.; 8]);
            assert_eq!(c.multiplier, 1.);
        }
    }
    let (_, mut lease) = setup(end, MAX_WORK);
    assert!(local_target(
        &g,
        &x,
        &b,
        &mut lease,
        &[1.],
        "relu",
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
    )
    .is_none());
}
#[test]
fn local_original_root_gain_unsafe_once_and_authenticated_source_guards() {
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let engine = LocalBackend::default();
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
        assert!(out[0].0 > 0.24 + shift && out[0].0 <= 0.25 + shift);
        assert_eq!(out[0].1.to_bits(), old.1.to_bits());
        assert_eq!(count(&engine.local_calls), 1);
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
        assert_eq!(count(&engine.local_calls), 1);
    }
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    // The true flag is an unchanged loader receipt of structurally authenticated
    // original f32 sources. Folded, custom-lowered and unauthenticated sources
    // all lack that receipt; a mere exact descriptor cannot override it.
    for mode in 0..3 {
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let mut c = config(&owner);
        match mode {
            0 => c.certified_root_cut_raw_source = false,
            1 => {
                c.verification_artifact_authority = VerificationArtifactAuthority::CertificateExport
            }
            _ => c.crown_backward_layers = Some(1),
        }
        let engine = LocalBackend::default();
        let mut out = [old];
        assert!(try_tighten(
            &c,
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
        assert_eq!(count(&engine.local_calls), 0);
        assert!(!owner.attempted.load(Ordering::SeqCst));
    }
}
#[test]
fn local_old_three_four_routes_preserve_results_and_pay_declined_advice() {
    for four in [false, true] {
        let (g, x, b) = if four {
            two_diamonds(0.)
        } else {
            cycle_fixture(0.)
        };
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let previous = QueryOwner::new(Some(end), true).unwrap();
        let new = QueryOwner::new(Some(end), true).unwrap();
        let old_engine = TripletBackend::default();
        let new_engine = Both::default();
        let mut a = [old];
        let mut c = [old];
        assert_eq!(
            try_tighten(
                &config(&previous),
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut a,
                &b,
                None,
                Some(&old_engine),
                Some(end)
            ),
            Some(0)
        );
        assert_eq!(
            try_tighten(
                &config(&new),
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut c,
                &b,
                None,
                Some(&new_engine),
                Some(end)
            ),
            Some(0)
        );
        assert_eq!(a, c);
        assert_eq!(count(&new_engine.local.local_calls), 0);
        let extra = (1 << 24) + 8192 + if four { 512 * 5 * 2 } else { 0 };
        assert_eq!(
            new.budget.receipt().unwrap().work_reserved,
            previous.budget.receipt().unwrap().work_reserved + extra
        );
        assert_eq!(
            count(&new_engine.old.triplet_calls),
            if four { 0 } else { 1 }
        );
        assert_eq!(
            count(&new_engine.old.joint.joint_calls),
            if four { 1 } else { 0 }
        );
    }
}
#[test]
fn local_entered_kernel_or_fold_refusal_never_retries_other_families() {
    for kernel_refusal in [false, true] {
        let (mut g, x, b) = fixture(0.);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        if kernel_refusal {
            let Layer::Linear(l) = g.node("pre").unwrap().layer() else {
                unreachable!()
            };
            let mut poisoned = l.clone();
            let mut weight = poisoned.weight().clone();
            weight[[0, 0]] = f32::NAN;
            poisoned.set_weight(weight).unwrap();
            g.nodes.test_insert(
                "pre".into(),
                crate::network::GraphNode::from_input("pre", Layer::Linear(poisoned)),
            );
        }
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let engine = Both::default();
        engine.local.refuse_local.store(1, Ordering::SeqCst);
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
        assert_eq!(
            count(&engine.local.local_calls),
            usize::from(!kernel_refusal)
        );
        assert_eq!(count(&engine.old.joint.ordinary.support_calls), 0);
        assert_eq!(count(&engine.old.joint.ordinary.cut_calls), 0);
        assert_eq!(count(&engine.old.joint.joint_calls), 0);
        assert_eq!(count(&engine.old.triplet_calls), 0);
        assert!(owner.attempted.load(Ordering::SeqCst));
    }
}
#[test]
fn local_preentry_storage_refusal_falls_back_with_same_spent_ledger_and_deadline() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let (plan, mut planning) = setup(end, MAX_WORK);
    planning.reserve_more_work(QUOTE_WORK).unwrap();
    let (name, pair, scale) = target(&g, &x, &b, &mut planning, &[1.], end).unwrap();
    assert!(local_target(&g, &x, &b, &mut planning, &[1.], name, end).is_some());
    let paid = plan.receipt().unwrap().work_reserved;
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
        RawRootCutLimits {
            max_nodes: MAX_NODES,
            max_scratch_bytes: 32 << 20,
            max_work: MAX_WORK,
        },
        end,
        &mut || true,
    )
    .unwrap();
    let (work, scratch, calls) = (q.work(), q.scratch_bytes(), q.candidate_count());
    let owner = QueryOwner::new(Some(end), true).unwrap();
    // Reserve ledger capacity only: no allocation. The old descriptor/fold fits
    // exactly, while the local immutable proof's extra4MiB cannot be admitted.
    let retained = owner
        .budget
        .lease(WorkRequest {
            scratch_bytes: MAX_SCRATCH - QUOTE_SCRATCH - FAMILY_SCRATCH - scratch,
            ..WorkRequest::default()
        })
        .unwrap();
    let engine = Both {
        single_only: true,
        ..Both::default()
    };
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
    assert_eq!(count(&engine.local.local_calls), 0);
    assert_eq!(count(&engine.old.joint.ordinary.support_calls), 1);
    assert_eq!(count(&engine.old.joint.ordinary.cut_calls), calls);
    assert_eq!(count(&engine.old.joint.joint_calls), 0);
    assert_eq!(
        owner.budget.receipt().unwrap().work_reserved,
        paid + QUOTE_WORK + work
    );
    assert!(out[0].0 >= old.0);
    assert_eq!(out[0].1.to_bits(), old.1.to_bits());
    assert!(owner.attempted.load(Ordering::SeqCst));
    drop(retained);
}

#[test]
fn local_raw_bn_original_source_owner_success_and_uncertainty_refuse_honestly() {
    use crate::layers::{BatchNormLayer, ReLULayer};
    use crate::network::GraphNode;
    use ndarray::arr1;
    let (mut g, x, _) = fixture(0.);
    let mut bn = BatchNormLayer::new(
        &arr1(&[1.; 9]).into_dyn(),
        &arr1(&[0.; 9]).into_dyn(),
        &arr1(&[0.; 9]).into_dyn(),
        &arr1(&[1.; 9]).into_dyn(),
        0.,
    )
    .unwrap();
    // This authored BN is exactly y=(x-0)/sqrt(1+0)*1+0=x.
    // Zero radii are proved by this exact formula, not by discarding unknowns.
    bn.scale_err.fill(0.);
    bn.bias_err.fill(0.);
    g.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn.clone()),
        vec!["pre".into()],
    ));
    g.nodes.test_insert(
        "relu".into(),
        GraphNode::new("relu", Layer::ReLU(ReLULayer), vec!["bn".into()]),
    );
    let b = g.collect_node_bounds(&x).unwrap();
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let engine = LocalBackend::default();
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
    assert!(out[0].0 > 0.24 && out[0].0 <= 0.25);
    assert_eq!(out[0].1.to_bits(), old.1.to_bits());
    assert_eq!(count(&engine.local_calls), 1);
    assert!(!engine.provides_bounded_support4());
    for malformed in [false, true] {
        let mut guarded = g.clone();
        let mut uncertain = bn.clone();
        uncertain
            .scale_err
            .fill(if malformed { f32::NAN } else { 1. / 1024. });
        uncertain.bias_err.fill(1. / 64.);
        guarded.nodes.test_insert(
            "bn".into(),
            GraphNode::new("bn", Layer::BatchNorm(uncertain), vec!["pre".into()]),
        );
        // A wider complete reference is safe for every declared nonzero-radius
        // source. Malformed values are negative controls only, never authority.
        let refs = if malformed {
            b.clone()
        } else {
            guarded.collect_node_bounds(&x).unwrap()
        };
        let old = (-100., 100.);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let engine = LocalBackend::default();
        let mut refused = [old];
        assert!(try_tighten(
            &config(&owner),
            &guarded,
            &x,
            &[vec![1.]],
            &[0.],
            &mut refused,
            &refs,
            None,
            Some(&engine),
            Some(end)
        )
        .is_none());
        assert_eq!(refused, [old]);
        if malformed {
            assert_eq!(count(&engine.local_calls), 0);
        } else {
            // Local certificate carries the radius correctly; the exact-only
            // test oracle explicitly refuses the charged nonzero full descriptor.
            assert_eq!(count(&engine.local_calls), 1);
        }
    }
}
