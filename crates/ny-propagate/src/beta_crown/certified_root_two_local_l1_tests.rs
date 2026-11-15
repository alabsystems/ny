// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::network::raw_root_cut::raw_local_l1_root_cut_test_support::fixture as four_diamonds;
use crate::network::raw_root_cut::raw_triplet_root_cut_test_support::{
    cycle_fixture, TripletBackend,
};
use crate::network::raw_root_cut::raw_two_local_l1_root_cut_test_support::{
    baseline, fixture, BundleBackend, SingleBackend,
};
use crate::network::raw_root_cut::raw_two_pair_root_cut_test_support::fixture as two_diamonds;
use ny_core::{
    GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment,
    ResidentLocalL1LowerCutCarrier, ResidentLowerCutCarrier, ResidentTripletLowerCutCarrier,
    ResidentTwoLocalL1LowerCutCarrier, ResidentTwoPairLowerCutCarrier,
};
use std::sync::atomic::AtomicUsize;

#[path = "certified_root_two_local_l1/small_pool_tests.rs"]
mod small_pool_tests;

#[path = "certified_root_two_local_l1/low_arity_tests.rs"]
mod low_arity_tests;

struct WithFallback {
    enabled: bool,
    local_enabled: bool,
    bundle: BundleBackend,
    local: SingleBackend,
    old: TripletBackend,
}
impl WithFallback {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            local_enabled: true,
            bundle: BundleBackend::default(),
            local: SingleBackend::default(),
            old: TripletBackend::default(),
        }
    }
    fn old_calls(&self) -> usize {
        count(&self.local.local_calls)
            + count(&self.old.triplet_calls)
            + count(&self.old.joint.joint_calls)
            + count(&self.old.joint.ordinary.cut_calls)
            + count(&self.old.joint.ordinary.support_calls)
    }
}
impl GemmEngine for WithFallback {
    fn gemm_f32(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        panic!("owner routing must not add ordinary contractions")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for WithFallback {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        panic!("owner routing must not add ordinary CROWN")
    }
    fn provides_bounded_two_local_l1_lower_cut(&self) -> bool {
        self.enabled
    }
    fn provides_bounded_local_l1_lower_cut(&self) -> bool {
        self.local_enabled
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
    fn provides_bounded_triplet_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTwoLocalL1LowerCutCarrier,
        d: Instant,
    ) -> ny_core::Result<f32> {
        self.bundle
            .crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(s, q, l, u, c, d)
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
fn output_weights(g: &mut GraphNetwork, values: &[f32]) {
    let Layer::Linear(l) = g.node("out").unwrap().layer() else {
        unreachable!()
    };
    let mut l = l.clone();
    l.set_weight(ndarray::Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap())
        .unwrap();
    g.nodes.test_insert(
        "out".into(),
        crate::network::GraphNode::new("out", Layer::Linear(l), vec!["relu".into()]),
    );
}

#[test]
fn two_local_ready_independent_capability_and_cpu_guard_preserve_custody() {
    let old = WithFallback::new(false);
    let bundle = BundleBackend::default();
    let mut gets = 0;
    let m = select_ready_methods(Some(&old), false, || {
        gets += 1;
        [Some(&bundle), None, None]
    })
    .unwrap();
    assert_eq!(gets, 1);
    assert!(std::ptr::eq(
        std::ptr::from_ref(m.two_local_l1.unwrap()).cast::<()>(),
        (&raw const bundle).cast::<()>()
    ));
    assert!(std::ptr::eq(
        std::ptr::from_ref(m.local_l1.unwrap()).cast::<()>(),
        (&raw const old).cast::<()>()
    ));
    let m = select_ready_methods(Some(&bundle), false, || [None, None, None]).unwrap();
    assert!(
        m.two_local_l1.is_some()
            && m.local_l1.is_none()
            && m.single.is_none()
            && m.joint.is_none()
            && m.triplet.is_none()
    );
    assert!(!bundle.provides_bounded_support4());
    assert!(raw_load_opportunity(Some(&bundle), false, false));
    assert!(!raw_load_opportunity(Some(&bundle), false, true));
    assert!(select_ready_methods(Some(&bundle), true, || panic!("CPU before getters")).is_none());
    assert!(
        select_ready_methods(Some(&WithFallback::new(true)), false, || panic!(
            "all ready locally"
        ))
        .is_some()
    );
    assert_eq!(count(&bundle.bundle_calls), 0);
    assert_eq!(old.old_calls(), 0);
}

#[test]
fn two_local_paid_top16_preserves_first8_and_splits_before_canonical_sort() {
    let (mut g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    for descending in [false, true] {
        let mut weights = vec![1.; 17];
        for i in 0..16 {
            weights[i] = -2.
                * if descending {
                    (16 - i) as f32
                } else {
                    (i + 1) as f32
                };
        }
        output_weights(&mut g, &weights);
        let (_, mut previous) = setup(end, MAX_WORK);
        let first = local_target(&g, &x, &b, &mut previous, &[1.], "relu", end).unwrap();
        let (budget, mut lease) = setup(end, MAX_WORK);
        let mut o = AdmissionObservation::default();
        let c = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut o).unwrap();
        assert_eq!(c.active, [8, 8]);
        assert_eq!(c.neurons[0], first.neurons);
        assert_eq!(c.weights[0], first.weights);
        assert_eq!(c.multipliers[0].to_bits(), first.multiplier.to_bits());
        assert_eq!(c.multipliers, [16., 8.]);
        assert_eq!(
            c.neurons[0],
            if descending {
                [0, 1, 2, 3, 4, 5, 6, 7]
            } else {
                [8, 9, 10, 11, 12, 13, 14, 15]
            }
        );
        assert!(o.scan_started && o.scan_complete);
        assert_eq!(
            (o.crossing, o.negative_crossing, o.groups),
            (16, 16, [8, 8])
        );
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            (1 << 24) + 8192 + 1024 * 17 * 2
        );
    }
    let mut weights = vec![1.; 17];
    weights[..11].fill(-2.);
    output_weights(&mut g, &weights);
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut o = AdmissionObservation::default();
    let c = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut o).unwrap();
    assert_eq!(c.active, [8, 3]);
    assert_eq!(c.neurons[0], [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(c.neurons[1], [8, 9, 10, 0, 0, 0, 0, 0]);
    assert_eq!(c.weights[1], [1., 1., 1., 0., 0., 0., 0., 0.]);
    assert_eq!(c.multipliers, [1., 1.]);
}

#[test]
fn two_local_admission_counts_distinguish_complete_insufficient_and_partial_scans() {
    let (mut g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let fixed = (1 << 24) + 8192;
    let scan = 1024 * 17 * 2;
    for (limit, spent) in [(fixed - 1, 0), (fixed + scan - 1, fixed)] {
        let (budget, mut lease) = setup(end, limit);
        let mut o = AdmissionObservation::default();
        assert!(bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut o).is_none());
        assert!(!o.scan_started && !o.scan_complete);
        assert_eq!(o.negative_crossing, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, spent);
    }
    let mut weights = vec![1.; 17];
    // A common Q8 scale preserves the existing eight-coordinate route.
    weights[..8].fill(-2.);
    output_weights(&mut g, &weights);
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut o = AdmissionObservation::default();
    assert!(bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut o).is_none());
    assert!(o.scan_started && o.scan_complete);
    assert_eq!(o.reason, "single-local-representable");
    assert_eq!((o.crossing, o.negative_crossing, o.groups), (16, 8, [8, 0]));
    weights[5] = f32::NAN;
    output_weights(&mut g, &weights);
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut o = AdmissionObservation::default();
    assert!(bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut o).is_none());
    assert!(o.scan_started && !o.scan_complete);
    assert_eq!(o.reason, "source-values");
    assert_eq!((o.crossing, o.negative_crossing, o.groups), (6, 5, [0, 0]));
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut o = AdmissionObservation::default();
    assert!(bundle_target(
        &g,
        &x,
        &b,
        &mut lease,
        &[1.],
        "relu",
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        &mut o
    )
    .is_none());
    assert!(!o.scan_started && !o.scan_complete);
    assert_eq!(o.reason, "deadline");
    assert!(
        size_of::<AdmissionObservation>()
            + 2 * size_of::<two_local_l1::BundleChoice>()
            + size_of::<[(usize, f64, f64); 16]>()
            + 2 * size_of::<[(usize, f64, f64); 8]>()
            + 2048
            <= FAMILY_SCRATCH
    );
}

#[test]
fn two_local_original_owner_positive_unsafe_once_and_no_support_dependency() {
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let engine = BundleBackend::default();
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
        assert_eq!(count(&engine.bundle_calls), 1);
        assert!(!engine.provides_bounded_support4());
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
        assert_eq!(count(&engine.bundle_calls), 1);
    }
}

#[test]
fn two_local_declined_scan_preserves_old_routes_and_exact_nonrefunded_debit() {
    for active in [3, 4, 8] {
        let (g, x, b) = match active {
            3 => cycle_fixture(0.),
            4 => two_diamonds(0.),
            _ => four_diamonds(0.),
        };
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let prior = QueryOwner::new(Some(end), true).unwrap();
        let current = QueryOwner::new(Some(end), true).unwrap();
        let a = WithFallback::new(false);
        let z = WithFallback::new(true);
        let mut before = [old];
        let mut after = [old];
        assert_eq!(
            try_tighten(
                &config(&prior),
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut before,
                &b,
                None,
                Some(&a),
                Some(end)
            ),
            Some(0)
        );
        assert_eq!(
            try_tighten(
                &config(&current),
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut after,
                &b,
                None,
                Some(&z),
                Some(end)
            ),
            Some(0)
        );
        assert_eq!(before, after);
        assert_eq!(count(&z.bundle.bundle_calls), 0);
        assert_eq!(a.old_calls(), z.old_calls());
        let width = b.get("pre").unwrap().len() as u64;
        let extra = BUNDLE_OBSERVATION_WORK + (1 << 24) + 8192 + 1024 * width * 2;
        assert_eq!(
            current.budget.receipt().unwrap().work_reserved,
            prior.budget.receipt().unwrap().work_reserved + extra
        );
        assert_eq!(current.deadline, prior.deadline);
        assert!(current.attempted.load(Ordering::SeqCst));
    }
}

#[test]
fn two_local_entered_kernel_or_fold_refusal_never_retries_old_methods() {
    for kernel_refusal in [false, true] {
        let (mut g, x, b) = fixture(0.);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        if kernel_refusal {
            let Layer::Linear(l) = g.node("pre").unwrap().layer() else {
                unreachable!()
            };
            let mut l = l.clone();
            let mut w = l.weight().clone();
            w[[8, 0]] = f32::NAN;
            l.set_weight(w).unwrap();
            g.nodes.test_insert(
                "pre".into(),
                crate::network::GraphNode::from_input("pre", Layer::Linear(l)),
            );
        }
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
            Some(end)
        )
        .is_none());
        assert_eq!(out, [old]);
        assert_eq!(engine.old_calls(), 0);
        assert_eq!(
            count(&engine.bundle.bundle_calls),
            usize::from(!kernel_refusal)
        );
        assert!(owner.attempted.load(Ordering::SeqCst));
    }
}

#[test]
fn two_local_source_guards_and_insufficient_setup_never_start_advice_or_numeric_work() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
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
        let engine = WithFallback::new(true);
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
        assert_eq!(engine.old_calls() + count(&engine.bundle.bundle_calls), 0);
        assert!(!owner.attempted.load(Ordering::SeqCst));
        assert_eq!(owner.budget.receipt().unwrap().work_reserved, 0);
    }
    // Atomic setup admission cannot spend the old amount and silently omit the
    // new observation debit. The owner attempt remains spent on refusal.
    let owner = Arc::new(QueryOwner {
        enabled: true,
        attempted: AtomicBool::new(false),
        deadline: end,
        budget: WorkBudget::new(
            Some(end),
            MAX_SCRATCH,
            0,
            QUOTE_WORK + BUNDLE_OBSERVATION_WORK - 1,
            1,
            Duration::ZERO,
        )
        .unwrap(),
    });
    let engine = WithFallback::new(true);
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
    assert_eq!(engine.old_calls() + count(&engine.bundle.bundle_calls), 0);
    assert_eq!(owner.budget.receipt().unwrap().work_reserved, 0);
    assert!(owner.attempted.load(Ordering::SeqCst));
}

/// Four diamonds/eight coordinates at each scale, interleaved by neuron index.
/// The original function is unchanged by scaling the weak coordinates by1024
/// and their output coefficients by1/1024. Every original input participates.
fn mixed_scale_fixture(
    shift: f32,
    weak_scale: f32,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let (mut g, x, _) = fixture(shift);
    let Layer::Linear(l) = g.node("pre").unwrap().layer() else {
        unreachable!()
    };
    let original = l.weight().clone();
    let mut scaled = original.clone();
    for i in 0..16 {
        let (row, scale) = if i % 2 == 0 {
            (i / 2, 1.)
        } else {
            (8 + i / 2, weak_scale)
        };
        for j in 0..16 {
            scaled[[i, j]] = original[[row, j]] * scale;
        }
    }
    let mut l = l.clone();
    l.set_weight(scaled).unwrap();
    g.nodes.test_insert(
        "pre".into(),
        crate::network::GraphNode::from_input("pre", Layer::Linear(l)),
    );
    let mut out = vec![1.; 17];
    for (i, value) in out[..16].iter_mut().enumerate() {
        *value = if i % 2 == 0 { -2. } else { -2. / weak_scale };
    }
    output_weights(&mut g, &out);
    let mut b = g.collect_node_bounds(&x).unwrap();
    // Supply exact authored-source boxes, independently justified by the two
    // nonzero dyadic coefficients per row and the stable sum. This makes the
    // equal-gap rank tie exact without claiming a particular collector's
    // outward rounding will reproduce that tie in a future CLI fixture.
    let mut lo = vec![0.; 17];
    let mut hi = vec![0.; 17];
    for i in 0..16 {
        hi[i] = if i % 2 == 0 { 1. } else { weak_scale };
        lo[i] = -hi[i];
    }
    lo[16] = 9.;
    hi[16] = 25.;
    let pre = BoundedTensor::new(
        ndarray::Array1::from_vec(lo).into_dyn(),
        ndarray::Array1::from_vec(hi.clone()).into_dyn(),
    )
    .unwrap();
    let mut post_lo = vec![0.; 17];
    post_lo[16] = 9.;
    let post = BoundedTensor::new(
        ndarray::Array1::from_vec(post_lo).into_dyn(),
        ndarray::Array1::from_vec(hi).into_dyn(),
    )
    .unwrap();
    b.insert("pre".into(), pre);
    b.insert("relu".into(), post);
    (g, x, b)
}

#[test]
fn two_local_window_rescue_mixed_scale_original_source_unsafe_and_once_only() {
    // Independent integer corner proof: each diamond has abs(u+v)+abs(u-v)=2
    // on every corner, hence convexity bounds its unscaled L1 by1 on the box.
    // Eight independent copies reach8. Exact binary scaling cancels in F.
    for u in [-1i32, 1] {
        for v in [-1i32, 1] {
            assert_eq!((u + v).abs() + (u - v).abs(), 2);
        }
    }
    assert_eq!(33i32 - 4 * 8, 1); // 4*true_min
    for shift in [0., -0.5] {
        let (g, x, b) = mixed_scale_fixture(shift, 1024.);
        let end = Instant::now() + Duration::from_secs(30);
        let (budget, mut lease) = setup(end, MAX_WORK);
        let mut obs = AdmissionObservation::default();
        let c = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).unwrap();
        assert!(obs.scan_complete);
        assert_eq!(
            (obs.crossing, obs.negative_crossing, c.active),
            (16, 16, [8, 8])
        );
        assert_eq!(
            c.neurons,
            [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
        );
        assert_eq!(c.weights, [[1.; 8]; 2]);
        assert_eq!(c.multipliers, [1., 1. / 1024.]);
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            (1 << 24) + 8192 + 1024 * 17 * 2 + two_local_l1::WINDOW_RESCUE_WORK
        );
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= -7.75 + shift && old.0 > -7.76 + shift);
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
        assert!(out[0].0 > 0.24 + shift && out[0].0 <= 0.25 + shift);
        assert_eq!(out[0].1.to_bits(), old.1.to_bits());
        assert_eq!(count(&engine.bundle.bundle_calls), 1);
        assert_eq!(engine.old_calls(), 0);
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
        assert_eq!(count(&engine.bundle.bundle_calls), 1);
    }
}

#[test]
fn two_local_window_rescue_keeps_valid_q8_bits_and_old_paid_path() {
    // Exact threshold: round(256/512)=1 remains valid, even though each old
    // group mixes scales. It must not enter the new selection rescue.
    let (boundary_g, boundary_x, boundary_b) = mixed_scale_fixture(0., 512.);
    let boundary_end = Instant::now() + Duration::from_secs(30);
    let (boundary_budget, mut boundary_lease) = setup(boundary_end, MAX_WORK);
    let mut boundary_obs = AdmissionObservation::default();
    let boundary = bundle_target(
        &boundary_g,
        &boundary_x,
        &boundary_b,
        &mut boundary_lease,
        &[1.],
        "relu",
        boundary_end,
        &mut boundary_obs,
    )
    .unwrap();
    assert_eq!(
        boundary.neurons,
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    );
    assert_eq!(
        boundary.weights,
        [[1., 1. / 256., 1., 1. / 256., 1., 1. / 256., 1., 1. / 256.]; 2]
    );
    assert_eq!(boundary.multipliers, [1., 1.]);
    assert_eq!(
        boundary_budget.receipt().unwrap().work_reserved,
        (1 << 24) + 8192 + 1024 * 17 * 2
    );
    let (mut g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    for small in [1., 1. / 128., 1. / 256., 1. / 512.] {
        let mut values = vec![1.; 17];
        for i in 0..16 {
            values[i] = -2. * if i % 2 == 0 { 1. } else { small };
        }
        output_weights(&mut g, &values);
        let (budget, mut lease) = setup(end, MAX_WORK);
        let mut obs = AdmissionObservation::default();
        let got = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).unwrap();
        // Independent literal old ranking/split/normalization for this source:
        // large-gap channels precede small ones, with original-index tie order.
        let mut ranked = (0usize..16).collect::<Vec<_>>();
        ranked.sort_by(|&a, &z| values[a].partial_cmp(&values[z]).unwrap().then(a.cmp(&z)));
        for group in 0..2 {
            let mut ids = ranked[group * 8..group * 8 + 8].to_vec();
            let max = ids
                .iter()
                .map(|&i| -f64::from(values[i]))
                .fold(0., f64::max);
            ids.sort_unstable();
            for (j, &i) in ids.iter().enumerate() {
                assert_eq!(got.neurons[group][j], i);
                let q = (256. * -f64::from(values[i]) / max).round() as f32 / 256.;
                assert_eq!(got.weights[group][j].to_bits(), q.to_bits());
            }
            assert_eq!(
                got.multipliers[group].to_bits(),
                ((max * 0.5) as f32).to_bits()
            );
        }
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            (1 << 24) + 8192 + 1024 * 17 * 2
        );
    }
}

#[test]
fn two_local_window_rescue_exact_extra_work_deadline_and_no_compatible_pair() {
    let (g, x, b) = mixed_scale_fixture(0., 1024.);
    let end = Instant::now() + Duration::from_secs(30);
    let ordinary = (1 << 24) + 8192 + 1024 * 17 * 2;
    let rescue = two_local_l1::WINDOW_RESCUE_WORK;
    for (cap, accepted, spent) in [
        (ordinary + rescue - 1, false, ordinary),
        (ordinary + rescue, true, ordinary + rescue),
    ] {
        let (budget, mut lease) = setup(end, cap);
        let mut obs = AdmissionObservation::default();
        assert_eq!(
            bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).is_some(),
            accepted
        );
        assert_eq!(budget.receipt().unwrap().work_reserved, spent);
        assert!(obs.scan_complete);
        if !accepted {
            assert_eq!((obs.stage, obs.reason), ("window-rescue", "window-work"));
        }
    }
    let mut ranked = [(0usize, 1f64, 0f64); 16];
    for (i, item) in ranked.iter_mut().enumerate() {
        *item = (
            i,
            1.,
            if i < 6 {
                2.
            } else if i < 11 {
                2. / 1024.
            } else {
                2. / 1_048_576.
            },
        );
    }
    let (budget, mut lease) = setup(end, rescue);
    let mut obs = AdmissionObservation::default();
    assert!(two_local_l1::rescue_windows(&ranked, 16, &mut lease, end, &mut obs).is_none());
    assert_eq!(obs.reason, "window-no-compatible-groups");
    assert_eq!(budget.receipt().unwrap().work_reserved, rescue);
    let (budget, mut lease) = setup(end, rescue);
    let mut obs = AdmissionObservation::default();
    assert!(two_local_l1::rescue_windows(
        &ranked,
        16,
        &mut lease,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        &mut obs
    )
    .is_none());
    assert_eq!(obs.reason, "deadline");
    assert_eq!(budget.receipt().unwrap().work_reserved, rescue);
    assert!(two_local_l1::rescue_storage_bytes().unwrap() <= FAMILY_SCRATCH);
}

#[test]
fn two_local_window_rescue_drops_only_declared_rows_and_rejects_other_failures() {
    // Three separated magnitude clusters, cardinalities5/8/3. The best legal
    // result covers the middle8 and upper5, explicitly leaving3 unselected.
    let end = Instant::now() + Duration::from_secs(30);
    let mut ranked = [(0usize, 1f64, 0f64); 16];
    for (i, item) in ranked.iter_mut().enumerate() {
        *item = (
            i,
            1.,
            if i < 5 {
                2.
            } else if i < 13 {
                2. / 1024.
            } else {
                2. / 1_048_576.
            },
        );
    }
    let (_, mut lease) = setup(end, two_local_l1::WINDOW_RESCUE_WORK);
    let mut obs = AdmissionObservation::default();
    let got = two_local_l1::rescue_windows(&ranked, 16, &mut lease, end, &mut obs).unwrap();
    assert_eq!(got.active, [8, 5]);
    assert_eq!(
        got.neurons,
        [[5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 4, 0, 0, 0]]
    );
    assert_eq!(got.weights, [[1.; 8], [1., 1., 1., 1., 1., 0., 0., 0.]]);
    let (mut g, x, b) = fixture(0.);
    let mut values = vec![-3_000_000_000.; 17];
    values[16] = 1.;
    output_weights(&mut g, &values);
    let (budget, mut lease) = setup(end, MAX_WORK);
    let mut obs = AdmissionObservation::default();
    assert!(bundle_target(&g, &x, &b, &mut lease, &[10.], "relu", end, &mut obs).is_none());
    assert!(obs.scan_complete);
    assert_eq!(
        (obs.stage, obs.reason),
        ("normalization", "q8-or-multiplier")
    );
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        (1 << 24) + 8192 + 1024 * 17 * 2
    );
}

#[test]
fn two_local_window_rescue_entered_refusal_never_salvages_another_family() {
    let (g, x, b) = mixed_scale_fixture(0., 1024.);
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
        Some(end)
    )
    .is_none());
    assert_eq!(out, [old]);
    assert_eq!(count(&engine.bundle.bundle_calls), 1);
    assert_eq!(engine.old_calls(), 0);
    assert!(owner.attempted.load(Ordering::SeqCst));
}

/// Three independent original frustrated cycles. Exact authored boxes make
/// the equal gap ordering explicit; this is not a collector-tightening claim.
fn nine_cycle_fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::GraphNode;
    use ndarray::{arr1, Array1, Array2};
    let mut w = Array2::<f32>::zeros((10, 9));
    for k in 0..3 {
        let j = 3 * k;
        w[[j, j]] = 0.5;
        w[[j, j + 1]] = 0.5;
        w[[j + 1, j + 1]] = 0.5;
        w[[j + 1, j + 2]] = 0.5;
        w[[j + 2, j + 2]] = 0.5;
        w[[j + 2, j]] = -0.5;
        w[[9, j + 1]] = 1.;
        w[[9, j + 2]] = 1.;
    }
    let mut bias = vec![0.; 10];
    bias[9] = 7.;
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(w, Some(Array1::from_vec(bias))).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    let mut final_w = vec![-2.; 10];
    final_w[9] = 1.;
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 10), final_w).unwrap(),
                Some(arr1(&[-0.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 9]).into_dyn(), arr1(&[1.; 9]).into_dyn()).unwrap();
    let mut b = g.collect_node_bounds(&x).unwrap();
    let mut lo = vec![-1.; 10];
    let mut hi = vec![1.; 10];
    lo[9] = 1.;
    hi[9] = 13.;
    b.insert(
        "pre".into(),
        BoundedTensor::new(
            Array1::from_vec(lo).into_dyn(),
            Array1::from_vec(hi.clone()).into_dyn(),
        )
        .unwrap(),
    );
    let mut post_lo = vec![0.; 10];
    post_lo[9] = 1.;
    b.insert(
        "relu".into(),
        BoundedTensor::new(
            Array1::from_vec(post_lo).into_dyn(),
            Array1::from_vec(hi).into_dyn(),
        )
        .unwrap(),
    );
    (g, x, b)
}

#[test]
fn two_local_nine_cycle_source_separation_positive_unsafe_and_once_only() {
    // Independent integer support proof: these are twice the original z.
    // Convexity of each selected L1 sum puts its maximum at a cube corner.
    let mut support2 = [0i32; 8];
    for u in [-1i32, 1] {
        for v in [-1i32, 1] {
            for w in [-1i32, 1] {
                let z2 = [u + v, v + w, w - u];
                assert_eq!(z2.iter().sum::<i32>(), 2 * (v + w));
                for mask in 0usize..8 {
                    let value = (0..3)
                        .filter(|i| mask & (1 << i) != 0)
                        .map(|i| z2[i].abs())
                        .sum();
                    support2[mask] = support2[mask].max(value);
                }
            }
        }
    }
    for (mask, value) in support2.iter().enumerate() {
        assert_eq!(
            *value,
            2 * (mask.count_ones() as i32 - if mask == 7 { 1 } else { 0 })
        );
    }
    let mut subsets = 0;
    for mask in 0usize..512 {
        let m = mask.count_ones() as usize;
        if !(3..=8).contains(&m) {
            continue;
        }
        subsets += 1;
        let c = (0..3).filter(|k| (mask >> (3 * k)) & 7 == 7).count() as i32;
        let bound2: i32 = (0..3).map(|k| support2[(mask >> (3 * k)) & 7]).sum();
        assert_eq!(bound2, 2 * (m as i32 - c));
        // Four times each complete ideal uniform half/full single-local fold.
        assert!(-11 + 4 * c <= -3);
        assert!(-11 + 2 * c <= -11 + 4 * c);
    }
    assert_eq!(subsets, 465);
    assert_eq!(25 - 4 * 6, 1); // 4*true_min; all u=v=w=1 attains it.
    for shift in [0., -0.5] {
        let (g, x, b) = nine_cycle_fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= -2.75 + shift && old.0 > -2.751 + shift);
        let single = SingleBackend::default();
        for lambda in [0.5, 1.] {
            let q = quote_raw_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                8,
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1.; 8],
                lambda,
                RawRootCutLimits {
                    max_nodes: 16,
                    max_scratch_bytes: 32 << 20,
                    max_work: 200_000_000,
                },
                end,
                &mut || true,
            )
            .unwrap();
            let out = evaluate_raw_local_l1_root_cut(q, &single, &mut || true).unwrap();
            assert!(out.bounds().0 <= -0.75 + shift);
            assert_eq!(out.bounds().1.to_bits(), old.1.to_bits());
        }
        assert_eq!(count(&single.local_calls), 2);
        let (_, mut lease) = setup(end, MAX_WORK);
        let mut obs = AdmissionObservation::default();
        let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).unwrap();
        assert_eq!(choice.active, [6, 3]);
        assert_eq!(
            choice.neurons,
            [[0, 1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]]
        );
        assert_eq!(
            choice.weights,
            [
                [1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 0., 0., 0., 0., 0.]
            ]
        );
        assert_eq!(choice.multipliers, [1., 1.]);
        let engine = WithFallback::new(true);
        let owner = QueryOwner::new(Some(end), true).unwrap();
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
        assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
        assert_eq!(count(&engine.local.local_calls), usize::from(shift < 0.));
        assert_eq!(engine.old_calls(), count(&engine.local.local_calls));
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

#[test]
fn two_local_nine_ten_preferred_lengths_padding_old_splits_and_exact_debit() {
    let (mut g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let scan = (1 << 24) + 8192 + 1024 * 17 * 2;
    for negatives in 9usize..=16 {
        let mut values = vec![1.; 17];
        values[..negatives].fill(-2.);
        output_weights(&mut g, &values);
        for cap in [scan - 1, scan] {
            let (budget, mut lease) = setup(end, cap);
            let mut obs = AdmissionObservation::default();
            let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs);
            if cap < scan {
                assert!(choice.is_none());
                assert_eq!(obs.reason, "selection-work");
                assert!(!obs.scan_started && !obs.scan_complete);
                assert_eq!(budget.receipt().unwrap().work_reserved, (1 << 24) + 8192);
                continue;
            }
            let choice = choice.unwrap();
            let first = if negatives == 9 {
                6
            } else if negatives == 10 {
                7
            } else {
                8
            };
            assert_eq!(choice.active, [first, negatives - first]);
            assert_eq!(obs.groups, choice.active);
            assert_eq!((obs.crossing, obs.negative_crossing), (16, negatives));
            assert!(obs.scan_complete);
            for group in 0..2 {
                let start = if group == 0 { 0 } else { first };
                for j in 0..8 {
                    let active = j < choice.active[group];
                    assert_eq!(choice.neurons[group][j], if active { start + j } else { 0 });
                    assert_eq!(
                        choice.weights[group][j].to_bits(),
                        if active { 1f32.to_bits() } else { 0 }
                    );
                }
                assert_eq!(choice.multipliers[group].to_bits(), 1f32.to_bits());
            }
            assert_eq!(budget.receipt().unwrap().work_reserved, scan);
        }
    }
}

#[test]
fn two_local_nine_ten_preserve_six_seven_single_routes_and_eight_positive() {
    // Existing retained test separately covers genuine3/4/8-family fixtures.
    // Here6/7 negative channels in the actual16-source test exercise the
    // unchanged local route with the extra bundle scan fully paid.
    for active in [6usize, 7, 8] {
        let (mut g, x, _) = fixture(0.);
        let mut values = vec![1.; 17];
        values[..active].fill(-2.);
        output_weights(&mut g, &values);
        let b = g.collect_node_bounds(&x).unwrap();
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        let mut results = Vec::new();
        let mut spends = Vec::new();
        let mut calls = Vec::new();
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
            assert_eq!(count(&engine.bundle.bundle_calls), 0);
            assert_eq!(count(&engine.local.local_calls), 1);
            assert_eq!(owner.deadline, end);
            results.push(out);
            spends.push(owner.budget.receipt().unwrap().work_reserved);
            calls.push(engine.old_calls());
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(calls[0], calls[1]);
        assert_eq!(
            spends[1],
            spends[0] + BUNDLE_OBSERVATION_WORK + (1 << 24) + 8192 + 1024 * 17 * 2
        );
        if active == 8 {
            assert!(results[1][0].0 > 0.24);
        }
    }
}

#[test]
fn two_local_nine_ten_rescue_windows_exact_budget_deadline_and_refusal() {
    let end = Instant::now() + Duration::from_secs(30);
    let work = two_local_l1::WINDOW_RESCUE_WORK;
    for count in [9usize, 10] {
        let mut ranked = [(0usize, 0f64, 0f64); 16];
        for (i, item) in ranked[..count].iter_mut().enumerate() {
            *item = (
                i,
                1.,
                if [1, 3, 5].contains(&i) {
                    2. / 1024.
                } else {
                    2.
                },
            );
        }
        // Actual owner advice crosses its preferred-normalization boundary:
        // exact source scaling balances all negative gap scores, so indices
        // remain rank ordered and the preferred6/7 rows mix both magnitudes.
        let (mut graph, input, _) = fixture(0.);
        let Layer::Linear(linear) = graph.node("pre").unwrap().layer() else {
            unreachable!()
        };
        let mut linear = linear.clone();
        let mut matrix = linear.weight().clone();
        for i in [1usize, 3, 5] {
            for j in 0..16 {
                matrix[[i, j]] *= 1024.;
            }
        }
        linear.set_weight(matrix).unwrap();
        graph.nodes.test_insert(
            "pre".into(),
            crate::network::GraphNode::from_input("pre", Layer::Linear(linear)),
        );
        let mut values = vec![1.; 17];
        for (i, value) in values[..count].iter_mut().enumerate() {
            *value = if [1, 3, 5].contains(&i) {
                -2. / 1024.
            } else {
                -2.
            };
        }
        output_weights(&mut graph, &values);
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let mut lo = vec![-1.; 17];
        let mut hi = vec![1.; 17];
        for i in [1usize, 3, 5] {
            lo[i] = -1024.;
            hi[i] = 1024.;
        }
        lo[16] = 9.;
        hi[16] = 25.;
        bounds.insert(
            "pre".into(),
            BoundedTensor::new(
                ndarray::Array1::from_vec(lo).into_dyn(),
                ndarray::Array1::from_vec(hi.clone()).into_dyn(),
            )
            .unwrap(),
        );
        let mut post_lo = vec![0.; 17];
        post_lo[16] = 9.;
        bounds.insert(
            "relu".into(),
            BoundedTensor::new(
                ndarray::Array1::from_vec(post_lo).into_dyn(),
                ndarray::Array1::from_vec(hi).into_dyn(),
            )
            .unwrap(),
        );
        let scan = (1 << 24) + 8192 + 1024 * 17 * 2;
        let (budget, mut lease) = setup(end, scan + work);
        let mut observed = AdmissionObservation::default();
        let actual = bundle_target(
            &graph,
            &input,
            &bounds,
            &mut lease,
            &[1.],
            "relu",
            end,
            &mut observed,
        )
        .unwrap();
        assert_eq!(actual.active, [count - 3, 3]);
        assert_eq!(actual.multipliers, [1., 1. / 1024.]);
        assert_eq!(actual.neurons[1], [1, 3, 5, 0, 0, 0, 0, 0]);
        assert!(observed.scan_complete && observed.failed_group.is_none());
        assert_eq!(
            (observed.negative_crossing, observed.groups),
            (count, actual.active)
        );
        assert_eq!(budget.receipt().unwrap().work_reserved, scan + work);
        for cap in [work - 1, work] {
            let (budget, mut lease) = setup(end, cap);
            let mut obs = AdmissionObservation::default();
            let got = two_local_l1::rescue_windows(&ranked, count, &mut lease, end, &mut obs);
            if cap < work {
                assert!(got.is_none());
                assert_eq!(obs.reason, "window-work");
                assert_eq!(budget.receipt().unwrap().work_reserved, 0);
                continue;
            }
            let got = got.unwrap();
            assert_eq!(got.active, [count - 3, 3]);
            assert_eq!(got.multipliers, [1., 1. / 1024.]);
            let expected: Vec<_> = (0..count).filter(|i| ![1, 3, 5].contains(i)).collect();
            assert_eq!(&got.neurons[0][..count - 3], expected.as_slice());
            assert_eq!(got.neurons[1], [1, 3, 5, 0, 0, 0, 0, 0]);
            for g in 0..2 {
                for i in 0..8 {
                    if i < got.active[g] {
                        assert_eq!(got.weights[g][i], 1.);
                    } else {
                        assert_eq!(got.neurons[g][i], 0);
                        assert_eq!(got.weights[g][i].to_bits(), 0);
                    }
                }
            }
            assert_eq!(budget.receipt().unwrap().work_reserved, work);
        }
        let (budget, mut lease) = setup(end, work);
        let mut obs = AdmissionObservation::default();
        assert!(two_local_l1::rescue_windows(
            &ranked,
            count,
            &mut lease,
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
            &mut obs
        )
        .is_none());
        assert_eq!(obs.reason, "deadline");
        assert_eq!(budget.receipt().unwrap().work_reserved, work);
        // Three separated clusters, none large enough for the first window.
        for (i, item) in ranked[..count].iter_mut().enumerate() {
            item.2 = if i < 4 {
                2.
            } else if i < 7 {
                2. / 1024.
            } else {
                2. / 1_048_576.
            };
        }
        let (budget, mut lease) = setup(end, work);
        let mut obs = AdmissionObservation::default();
        assert!(two_local_l1::rescue_windows(&ranked, count, &mut lease, end, &mut obs).is_none());
        assert_eq!(obs.reason, "window-no-compatible-groups");
        assert_eq!(budget.receipt().unwrap().work_reserved, work);
    }
    assert!(two_local_l1::rescue_storage_bytes().unwrap() <= FAMILY_SCRATCH);
}

#[test]
fn two_local_nine_cycle_entered_refusal_has_no_old_family_retry() {
    for kernel_refusal in [false, true] {
        let (mut g, x, b) = nine_cycle_fixture(-0.5);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        if kernel_refusal {
            let Layer::Linear(l) = g.node("pre").unwrap().layer() else {
                unreachable!()
            };
            let mut l = l.clone();
            let mut w = l.weight().clone();
            w[[6, 0]] = f32::NAN;
            l.set_weight(w).unwrap();
            g.nodes.test_insert(
                "pre".into(),
                crate::network::GraphNode::from_input("pre", Layer::Linear(l)),
            );
        }
        let engine = WithFallback::new(true);
        engine.bundle.refuse_bundle.store(1, Ordering::SeqCst);
        let owner = QueryOwner::new(Some(end), true).unwrap();
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
            count(&engine.bundle.bundle_calls),
            usize::from(!kernel_refusal)
        );
        assert_eq!(
            count(&engine.local.local_calls),
            usize::from(!kernel_refusal)
        );
        assert_eq!(engine.old_calls(), count(&engine.local.local_calls));
        assert!(owner.attempted.load(Ordering::SeqCst));
        assert_eq!(owner.deadline, end);
    }
}

/// A source-valid selected-row intersection, with unselected endpoint bits
/// retained exactly as the head contract requires. This does not execute the
/// collector: the fixture's exact extrema independently justify both maps.
fn nine_head_like_map(
    bounds: &HashMap<String, BoundedTensor>,
    tight: &[usize],
) -> HashMap<String, BoundedTensor> {
    use ndarray::Array1;
    let wide = f32::from_bits(1f32.to_bits() + 1);
    let mut lower = vec![-wide; 10];
    let mut upper = vec![wide; 10];
    lower[9] = 1.;
    upper[9] = 13.;
    for &i in tight {
        assert!(i < 9);
        lower[i] = -1.;
        upper[i] = 1.;
    }
    let mut out = bounds.clone();
    out.insert(
        "pre".into(),
        BoundedTensor::new(
            Array1::from_vec(lower).into_dyn(),
            Array1::from_vec(upper.clone()).into_dyn(),
        )
        .unwrap(),
    );
    let mut post_lower = vec![0.; 10];
    post_lower[9] = 1.;
    out.insert(
        "relu".into(),
        BoundedTensor::new(
            Array1::from_vec(post_lower).into_dyn(),
            Array1::from_vec(upper).into_dyn(),
        )
        .unwrap(),
    );
    out
}
fn intact_nine_partition(choice: &two_local_l1::BundleChoice) -> bool {
    choice.active == [6, 3]
        && (0usize..3).any(|cycle| {
            let triple = [3 * cycle, 3 * cycle + 1, 3 * cycle + 2];
            let complement: Vec<_> = (0usize..9).filter(|i| !triple.contains(i)).collect();
            choice.neurons[1][..3] == triple
                && choice.neurons[0][..6] == complement
                && choice.neurons[0][6..].iter().all(|i| *i == 0)
                && choice.neurons[1][3..].iter().all(|i| *i == 0)
        })
}

#[test]
fn two_local_geometry_recovers_actual_perturbed_map_safe_unsafe_and_upper_bits() {
    for shift in [0., -0.5] {
        let (g, x, original) = nine_cycle_fixture(shift);
        for tight in [vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 1, 2, 3]] {
            let b = nine_head_like_map(&original, &tight);
            let end = Instant::now() + Duration::from_secs(30);
            let old = baseline(&g, &x, &b, end);
            let (_, mut lease) = setup(end, MAX_WORK);
            let mut obs = AdmissionObservation::default();
            let selection = two_local_l1::bundle_selection(
                &g,
                &x,
                &b,
                &mut lease,
                &[1.],
                "relu",
                end,
                &mut obs,
            )
            .unwrap();
            assert_eq!(selection.count, 9);
            assert!(!intact_nine_partition(&selection.choice));
            let q = quote_raw_two_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                selection.choice.active,
                selection.choice.neurons,
                selection.choice.weights,
                selection.choice.multipliers,
                RawRootCutLimits {
                    max_nodes: 16,
                    max_scratch_bytes: 32 << 20,
                    max_work: 200_000_000,
                },
                end,
                &mut || true,
            )
            .unwrap();
            let bad = BundleBackend::default();
            let candidate = evaluate_raw_two_local_l1_root_cut(q, &bad, &mut || true).unwrap();
            assert!(candidate.bounds().0 <= -1.75 + shift);
            assert_eq!(candidate.bounds().1.to_bits(), old.1.to_bits());
            let source = two_local_l1::geometry::geometry_quote(
                &g, &x, &b, "relu", &mut lease, end, &mut obs,
            )
            .unwrap();
            let choice = source
                .evaluate(
                    &selection.ranked,
                    9,
                    &selection.choice,
                    &mut lease,
                    end,
                    &mut obs,
                )
                .unwrap();
            assert!(intact_nine_partition(&choice));
            let engine = WithFallback::new(true);
            let owner = QueryOwner::new(Some(end), true).unwrap();
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
            assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
            assert_eq!(count(&engine.local.local_calls), usize::from(shift < 0.));
            assert_eq!(engine.old_calls(), count(&engine.local.local_calls));
            assert!(try_tighten(
                &config(&owner),
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut [old],
                &b,
                None,
                Some(&engine),
                Some(end)
            )
            .is_none());
            assert_eq!(count(&engine.bundle.bundle_calls), usize::from(shift < 0.));
        }
    }
}

/// Expand each independent input into a disjoint dyadic mean. All1024 inputs
/// participate; the old certified node boxes remain valid for this identical
/// nine-variable image. This sizes the admission comparison without a model.
fn wide_nine_cycle_fixture() -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    use ndarray::{Array1, Array2};
    let (mut g, _, b) = nine_cycle_fixture(0.);
    let Layer::Linear(old) = g.node("pre").unwrap().layer() else {
        unreachable!()
    };
    let sizes = [256usize, 256, 128, 128, 64, 64, 64, 32, 32];
    let mut matrix = Array2::zeros((10, 1024));
    let mut start = 0;
    for (i, size) in sizes.iter().enumerate() {
        for row in 0..10 {
            for col in start..start + size {
                matrix[[row, col]] = old.weight()[[row, i]] / *size as f32;
            }
        }
        start += size;
    }
    assert_eq!(start, 1024);
    let bias = old.bias().cloned();
    let layer = crate::layers::LinearLayer::new(matrix, bias).unwrap();
    g.nodes.test_insert(
        "pre".into(),
        crate::network::GraphNode::from_input("pre", Layer::Linear(layer)),
    );
    let x = BoundedTensor::new(
        Array1::from_elem(1024, -1.).into_dyn(),
        Array1::from_elem(1024, 1.).into_dyn(),
    )
    .unwrap();
    (g, x, b)
}

#[test]
fn two_local_geometry_combined_preflight_preserves_paid_old_triplet_fallback() {
    use two_local_l1::geometry_work_fits;
    assert!(!geometry_work_fits(u64::MAX, u64::MAX, 1));
    let (g, x, b) = wide_nine_cycle_fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    // First measure only this fixture's deterministic static work ledger. This
    // is not runtime calibration and the numerical result is independently checked.
    let full = QueryOwner::new(Some(end), true).unwrap();
    // This retained control prices the one-fold backend-only capability;
    // the comparison upgrade has separate exact-budget controls below.
    let mut engine = WithFallback::new(true);
    engine.local_enabled = false;
    let mut out = [old];
    assert_eq!(
        try_tighten(
            &config(&full),
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
    let total = full.budget.receipt().unwrap().work_reserved;
    // The previously complete advice+fresh-fold budget is short by exactly1.
    // A preflight must decline BEFORE the E-dependent advice debit.
    let remaining = total - 1;
    let prior = QueryOwner::new(Some(end), true).unwrap();
    let mut old_engine = WithFallback::new(false);
    old_engine.local_enabled = false;
    let mut expected = [old];
    assert_eq!(
        try_tighten(
            &config(&prior),
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut expected,
            &b,
            None,
            Some(&old_engine),
            Some(end)
        ),
        Some(0)
    );
    let extra = BUNDLE_OBSERVATION_WORK + (1 << 24) + 8192 + 1024 * 10 * 2 + (1 << 20) + QUOTE_WORK;
    // The fixture must leave enough residual work for the retained triplet,
    // including every paid pre-entry bundle attempt. Derived total-1 alone
    // does not establish this because the triplet has a large fixed charge.
    assert!(remaining >= prior.budget.receipt().unwrap().work_reserved + extra);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let precharge = MAX_WORK - remaining;
    let _held = owner
        .budget
        .lease(WorkRequest {
            work: precharge,
            ..WorkRequest::default()
        })
        .unwrap();
    let mut fallback_engine = WithFallback::new(true);
    fallback_engine.local_enabled = false;
    let mut actual = [old];
    assert_eq!(
        try_tighten(
            &config(&owner),
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut actual,
            &b,
            None,
            Some(&fallback_engine),
            Some(end)
        ),
        Some(0)
    );
    assert_eq!(actual, expected);
    assert_eq!(actual[0].1.to_bits(), old.1.to_bits());
    assert_eq!(count(&fallback_engine.bundle.bundle_calls), 0);
    assert_eq!(count(&fallback_engine.old.triplet_calls), 1);
    assert_eq!(
        owner.budget.receipt().unwrap().work_reserved - precharge,
        prior.budget.receipt().unwrap().work_reserved + extra
    );
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut obs = AdmissionObservation::default();
    let selection =
        two_local_l1::bundle_selection(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs)
            .unwrap();
    let source =
        two_local_l1::geometry::geometry_quote(&g, &x, &b, "relu", &mut lease, end, &mut obs)
            .unwrap();
    let advice = source.work(9).unwrap();
    let execution = 12345;
    let exact = advice + QUOTE_WORK + execution;
    assert!(!geometry_work_fits(exact - 1, advice, execution));
    assert!(geometry_work_fits(exact, advice, execution));
    assert_eq!(selection.count, 9);
}

#[test]
fn two_local_geometry_entered_source_or_fold_refusal_never_retries_other_family() {
    for malformed_source in [false, true] {
        let (mut g, x, b) = nine_cycle_fixture(-0.5);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        if malformed_source {
            let Layer::Linear(l) = g.node("pre").unwrap().layer() else {
                unreachable!()
            };
            let mut l = l.clone();
            let mut w = l.weight().clone();
            w[[8, 8]] = f32::NAN;
            l.set_weight(w).unwrap();
            g.nodes.test_insert(
                "pre".into(),
                crate::network::GraphNode::from_input("pre", Layer::Linear(l)),
            );
        }
        let engine = WithFallback::new(true);
        if !malformed_source {
            engine.bundle.refuse_bundle.store(1, Ordering::SeqCst);
        }
        let owner = QueryOwner::new(Some(end), true).unwrap();
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
            count(&engine.bundle.bundle_calls),
            usize::from(!malformed_source)
        );
        assert_eq!(
            count(&engine.local.local_calls),
            usize::from(!malformed_source)
        );
        assert_eq!(engine.old_calls(), count(&engine.local.local_calls));
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
        assert_eq!(
            count(&engine.local.local_calls),
            usize::from(!malformed_source)
        );
        assert_eq!(engine.old_calls(), count(&engine.local.local_calls));
    }
}

#[test]
fn two_local_single3_owner_only_nine_ten_upgrades_and_exact_added_debit() {
    for negatives in [9usize, 10, 11, 16] {
        let (mut g, x, mut b) = if negatives == 9 {
            nine_cycle_fixture(0.)
        } else {
            fixture(0.)
        };
        if negatives != 9 {
            let mut values = vec![1.; 17];
            values[..negatives].fill(-2.);
            output_weights(&mut g, &values);
            b = g.collect_node_bounds(&x).unwrap();
        }
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        let (_, mut lease) = setup(end, MAX_WORK);
        let mut obs = AdmissionObservation::default();
        let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).unwrap();
        let q = quote_raw_two_local_l1_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "relu",
            choice.active,
            choice.neurons,
            choice.weights,
            choice.multipliers,
            RawRootCutLimits {
                max_nodes: MAX_NODES,
                max_scratch_bytes: MAX_SCRATCH,
                max_work: MAX_WORK,
            },
            end,
            &mut || true,
        )
        .unwrap();
        let once = q.work();
        let extra = if negatives <= 10 {
            let cap = RawRootCutLimits {
                max_nodes: MAX_NODES,
                max_scratch_bytes: MAX_SCRATCH,
                max_work: MAX_WORK,
            };
            let mut q = q.with_single3_comparison(cap).unwrap();
            assert!(q.try_enable_full_head_advice(cap));
            assert!(q.try_enable_head_box_floor(1., cap));
            q.work() - once
        } else {
            0
        };
        let mut spent = Vec::new();
        let mut actual = Vec::new();
        for comparison_capability in [false, true] {
            let mut engine = WithFallback::new(true);
            engine.local_enabled = comparison_capability;
            let owner = QueryOwner::new(Some(end), true).unwrap();
            let mut out = [old];
            let _ = try_tighten(
                &config(&owner),
                &g,
                &x,
                &[vec![1.]],
                &[1.],
                &mut out,
                &b,
                None,
                Some(&engine),
                Some(end),
            );
            assert_eq!(count(&engine.bundle.bundle_calls), 1);
            let expected = usize::from(comparison_capability && negatives <= 10);
            assert_eq!(count(&engine.local.local_calls), expected);
            assert_eq!(engine.old_calls(), expected);
            assert!(out[0].0 >= old.0);
            assert_eq!(out[0].1.to_bits(), old.1.to_bits());
            spent.push(owner.budget.receipt().unwrap().work_reserved);
            actual.push(out[0]);
        }
        assert_eq!(spent[1], spent[0] + extra);
        // The advice policy preserves its actual single3 endpoint; it need
        // not dominate an unexecuted bundle with the original multiplier.
        assert!(actual[1].0 >= old.0);
        if negatives > 10 {
            assert_eq!(actual[0], actual[1]);
        }
    }
    // A complete real source scan with103 eligible cells still retains top16.
    // It is not a nine/ten source merely because some future partition has3
    // cells. The bounded oracle deliberately refuses width104 after dispatch;
    // this control checks routing/call cardinality, not its numerical result.
    let mut g = GraphNetwork::new();
    let mut bias = vec![0.; 104];
    bias[103] = 2.;
    g.add_node(crate::network::GraphNode::from_input(
        "pre",
        Layer::Linear(
            crate::layers::LinearLayer::new(
                ndarray::Array2::from_elem((104, 1), 1.),
                Some(ndarray::Array1::from_vec(bias)),
            )
            .unwrap(),
        ),
    ));
    g.add_node(crate::network::GraphNode::new(
        "relu",
        Layer::ReLU(crate::layers::ReLULayer),
        vec!["pre".into()],
    ));
    let mut coefficients = vec![-2.; 104];
    coefficients[103] = 1.;
    g.add_node(crate::network::GraphNode::new(
        "out",
        Layer::Linear(
            crate::layers::LinearLayer::new(
                ndarray::Array2::from_shape_vec((1, 104), coefficients).unwrap(),
                None,
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(
        ndarray::arr1(&[-1.]).into_dyn(),
        ndarray::arr1(&[1.]).into_dyn(),
    )
    .unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    let end = Instant::now() + Duration::from_secs(60);
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut obs = AdmissionObservation::default();
    let selected =
        two_local_l1::bundle_selection(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs)
            .unwrap();
    assert_eq!(
        (selected.count, obs.negative_crossing, obs.scan_complete),
        (16, 103, true)
    );
    assert_eq!(selected.choice.active, [8, 8]);
    let engine = WithFallback::new(true);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let mut out = [(-256., 256.)];
    let original = out;
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
    assert_eq!(out, original);
    assert_eq!(
        (
            count(&engine.local.local_calls),
            count(&engine.bundle.bundle_calls)
        ),
        (0, 1)
    );
    assert_eq!(engine.old_calls(), 0);
}

#[test]
fn two_local_single3_owner_second_fold_refusal_keeps_original_bound_and_attempt() {
    let (g, x, b) = nine_cycle_fixture(-0.5);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let engine = WithFallback::new(true);
    engine.bundle.refuse_bundle.store(1, Ordering::SeqCst);
    let owner = QueryOwner::new(Some(end), true).unwrap();
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
        (
            count(&engine.local.local_calls),
            count(&engine.bundle.bundle_calls)
        ),
        (1, 1)
    );
    assert_eq!(engine.old_calls(), 1); // The predeclared single3, no prefix/family retry.
    assert!(owner.attempted.load(Ordering::SeqCst));
    let debit = owner.budget.receipt().unwrap().work_reserved;
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
    assert_eq!(owner.budget.receipt().unwrap().work_reserved, debit);
    assert_eq!(
        (
            count(&engine.local.local_calls),
            count(&engine.bundle.bundle_calls)
        ),
        (1, 1)
    );
}

#[test]
fn two_local_full_head_owner_one_work_short_uses_declared_one_prefix_portfolio() {
    let (g, x, b) = nine_cycle_fixture(-0.5);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let (_, mut lease) = setup(end, MAX_WORK);
    let mut obs = AdmissionObservation::default();
    let choice = bundle_target(&g, &x, &b, &mut lease, &[1.], "relu", end, &mut obs).unwrap();
    assert_eq!(
        (obs.negative_crossing, obs.scan_complete, choice.active),
        (9, true, [6, 3])
    );
    let cap = RawRootCutLimits {
        max_nodes: MAX_NODES,
        max_scratch_bytes: MAX_SCRATCH,
        max_work: MAX_WORK,
    };
    let mut q = quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        choice.active,
        choice.neurons,
        choice.weights,
        choice.multipliers,
        cap,
        end,
        &mut || true,
    )
    .unwrap()
    .with_single3_comparison(cap)
    .unwrap();
    assert!(q.try_enable_full_head_advice(cap));
    assert!(q.try_enable_head_box_floor(0., cap));
    let complete_work = q.work();
    assert_eq!(q.candidate_count(), 3);
    // Reconstruct the shape quote at exactly one work unit below the full
    // portfolio. Both source certificates remain required; only the declared
    // single3 prefix is dropped, with its complete charge removed.
    let mut reduced = quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        choice.active,
        choice.neurons,
        choice.weights,
        choice.multipliers,
        cap,
        end,
        &mut || true,
    )
    .unwrap()
    .with_single3_comparison(cap)
    .unwrap();
    let reduced_cap = RawRootCutLimits {
        max_work: complete_work - 1,
        ..cap
    };
    assert!(reduced.try_enable_full_head_advice(reduced_cap));
    assert!(reduced.try_enable_head_box_floor(0., reduced_cap));
    assert_eq!(reduced.candidate_count(), 2);
    let reduced_work = reduced.work();
    assert!(reduced_work < complete_work);
    let full = QueryOwner::new(Some(end), true).unwrap();
    let mut out = [old];
    let engine = WithFallback::new(true);
    let _ = try_tighten(
        &config(&full),
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
    assert_eq!(
        (
            count(&engine.local.local_calls),
            count(&engine.bundle.bundle_calls)
        ),
        (1, 1)
    );
    let total = full.budget.receipt().unwrap().work_reserved;
    assert!(total > complete_work);
    let remaining = total - 1;
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let precharge = MAX_WORK - remaining;
    let _held = owner
        .budget
        .lease(WorkRequest {
            work: precharge,
            ..WorkRequest::default()
        })
        .unwrap();
    let engine = WithFallback::new(true);
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
    assert_eq!(
        (
            count(&engine.local.local_calls),
            count(&engine.bundle.bundle_calls)
        ),
        (0, 1)
    );
    assert_eq!(engine.old_calls(), 0); // Only the declared bundle prefix, no family retry.
    assert_eq!(
        owner.budget.receipt().unwrap().work_reserved - precharge,
        total - complete_work + reduced_work
    );
    assert!(out[0].0 >= old.0);
    assert_eq!(out[0].1.to_bits(), old.1.to_bits());
}
