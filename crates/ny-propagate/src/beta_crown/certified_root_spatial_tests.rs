// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use crate::network::raw_root_cut::raw_two_local_l1_root_cut_test_support::exact_plain_rows;
use crate::network::GraphNode;
use ndarray::{arr1, Array1, Array2};
use ny_core::{GpuCrownLayer, GpuCrownResult, GpuResnetSegment, NyError, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[cfg(feature = "gpu-tests-cuda")]
#[path = "certified_root_spatial_device_tests.rs"]
mod device;

const N: usize = 9;
const EPS: f32 = 1. / 1024.;

fn fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    let k1 = ArrayD::from_elem(IxDyn(&[2 * N, 1, 1, 1]), 1.);
    let mut k2 = ArrayD::zeros(IxDyn(&[N + 1, 2 * N, 1, 1]));
    for i in 0..N {
        k2[[i, 2 * i, 0, 0]] = 1.;
        k2[[i, 2 * i + 1, 0, 0]] = -1.;
    }
    k2[[N, 0, 0, 0]] = 1.;
    let mut bias = Array1::zeros(N + 1);
    bias[N] = -2.;
    g.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(
            Conv2dLayer::new(k1, Some(Array1::from_elem(2 * N, 2.)), (1, 1), (0, 0)).unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".into()],
    ));
    g.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(Conv2dLayer::new(k2, Some(bias), (1, 1), (0, 0)).unwrap()),
        vec!["relu1".into()],
    ));
    g.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["conv2".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["relu2".into()],
    ));
    let mut weights = vec![-1.; N + 1];
    weights[N] = -EPS;
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, N + 1), weights).unwrap(),
                Some(arr1(&[0.25 + EPS + shift])),
            )
            .unwrap(),
        ),
        vec!["flat".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.),
    )
    .unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    Refuse,
    Count,
    Nan,
    Infinite,
    Inverted,
    Disjoint,
    Late,
}
struct Engine {
    calls: AtomicUsize,
    seen: Mutex<Vec<(usize, usize)>>,
    at: usize,
    fault: Fault,
    capacity: usize,
}
impl Engine {
    fn new(fault: Fault, at: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
            at,
            fault,
            capacity: 8,
        }
    }
}
impl GemmEngine for Engine {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("no ordinary GEMM retry")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for Engine {
    fn provides_sound_gpu_crown(&self) -> bool {
        true
    }
    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        self.capacity
    }
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("no ordinary CROWN retry")
    }
    fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        _: &[Vec<f32>],
        _: &[Vec<f32>],
        end: Instant,
    ) -> Result<GpuCrownResult> {
        assert!(Instant::now() < end);
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        self.seen.lock().unwrap().push((q.num_specs, q.current_dim));
        if call == self.at && matches!(self.fault, Fault::Refuse) {
            return Err(NyError::DeadlineExceeded("required fold refused".into()));
        }
        let mut got = exact_plain_rows(s, q, l, u, end)?;
        if call == self.at {
            match self.fault {
                Fault::Count => {
                    got.lower_bounds.pop();
                }
                Fault::Nan => got.lower_bounds[0] = f32::NAN,
                Fault::Infinite => got.upper_bounds[0] = f32::INFINITY,
                Fault::Inverted => {
                    got.lower_bounds[0] = 2.;
                    got.upper_bounds[0] = 1.;
                }
                Fault::Disjoint => {
                    got.lower_bounds[0] = 100.;
                    got.upper_bounds[0] = 101.;
                }
                Fault::Late => {
                    let left = end.saturating_duration_since(Instant::now());
                    assert!(left <= Duration::from_millis(250));
                    std::thread::sleep(left + Duration::from_millis(1));
                }
                _ => {}
            }
        }
        Ok(got)
    }
}

fn end() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn old(
    g: &GraphNetwork,
    x: &BoundedTensor,
    b: &HashMap<String, BoundedTensor>,
    end: Instant,
) -> (f32, f32) {
    let limits = ExtractionLimits {
        max_nodes: 128,
        max_caller_bytes: 256 << 20,
    };
    let full =
        extract_charged_raw_bn_chain(g, x, "out", b, b, None, limits, end, &mut || true).unwrap();
    let a = Arc::<[f32]>::from(vec![1.]);
    let z = Arc::<[f32]>::from(vec![0.]);
    let seed = GpuCrownSeed {
        lower_a: a.clone(),
        upper_a: a,
        lower_b: z.clone(),
        upper_b: z,
        num_specs: 1,
        current_dim: 1,
    };
    let v = exact_plain_rows(
        &full.segments,
        &seed,
        x.lower().as_slice().unwrap(),
        x.upper().as_slice().unwrap(),
        end,
    )
    .unwrap();
    (v.lower_bounds[0], v.upper_bounds[0])
}
fn quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
) -> Quote<'a> {
    let ranked = (0..N).map(|i| (i, 1.)).collect::<Vec<_>>();
    quote_rows(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "conv2",
        &ranked,
        8,
        MAX_WORK,
        MAX_SCRATCH,
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn spatial_feedback_complete_original_multibatch_source_closes_and_preserves_unsafe_shift() {
    // Original pairs are (x+2,x+2), so every difference is exactly zero.
    // The final unselected neuron is x and keeps its nonzero -EPS objective.
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = end();
        let old = old(&g, &x, &b, end);
        assert!((old.0 - (0.25 - N as f32 + shift)).abs() < 0.0001);
        let q = quote(&g, &x, &b, old, end);
        assert_eq!(q.selected, N);
        assert_eq!(q.calls(), 3);
        let engine = Engine::new(Fault::None, 0);
        let got = evaluate(&q, &engine, &mut || true).unwrap();
        assert!(got.lower > 0.249 + shift && got.lower <= 0.25 + shift);
        assert_eq!(got.prefix_calls, 2);
        assert_eq!(
            *engine.seen.lock().unwrap(),
            vec![(8, N + 1), (1, N + 1), (1, 1)]
        );
        assert!(b["conv2"].lower().as_slice().unwrap()[0] < -1.99);
        // Exact input endpoints independently retain the complete tiny term.
        for v in [-1f32, 0., 1.] {
            let truth = 0.25 + EPS + shift - EPS * v.max(0.);
            assert!(got.lower <= truth);
        }
    }
}

#[test]
fn spatial_feedback_every_required_fold_payload_and_each_poll_refuse_without_partial_publication() {
    let (g, x, b) = fixture(0.);
    let end = end();
    let old = old(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let original_lower = b["conv2"].lower().as_slice().unwrap().to_vec();
    for fault in [
        Fault::Refuse,
        Fault::Count,
        Fault::Nan,
        Fault::Infinite,
        Fault::Inverted,
        Fault::Disjoint,
    ] {
        for at in 1..=3 {
            let engine = Engine::new(fault, at);
            assert!(evaluate(&q, &engine, &mut || true).is_none());
            assert_eq!(engine.calls.load(Ordering::SeqCst), at);
        }
    }
    let mut polls = 0;
    assert!(evaluate(&q, &Engine::new(Fault::None, 0), &mut || {
        polls += 1;
        true
    })
    .is_some());
    for stop in 1..=polls {
        let mut n = 0;
        assert!(
            evaluate(&q, &Engine::new(Fault::None, 0), &mut || {
                n += 1;
                n < stop
            })
            .is_none(),
            "stop={stop}/{polls}"
        );
    }
    assert_eq!(
        b["conv2"].lower().as_slice().unwrap(),
        original_lower.as_slice()
    );
}

#[test]
fn spatial_feedback_quote_exact_floor_budgets_capacity_and_late_result_are_fail_closed() {
    let (g, x, b) = fixture(0.);
    let end = end();
    let old = old(&g, &x, &b, end);
    let make = |w, bytes| {
        quote_rows(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "conv2",
            &[(0, 1.)],
            8,
            w,
            bytes,
            end,
            &mut || true,
        )
    };
    let q = make(MAX_WORK, MAX_SCRATCH).unwrap();
    let (w, bytes) = (q.work(), q.bytes());
    assert!(make(w, bytes).is_some());
    assert!(make(w - 1, bytes).is_none());
    assert!(make(w, bytes - 1).is_none());
    for capacity in [0, 9] {
        let mut e = Engine::new(Fault::None, 0);
        e.capacity = capacity;
        assert!(evaluate(&q, &e, &mut || true).is_none());
        assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    }
    let late_end = Instant::now() + Duration::from_millis(100);
    let late = quote(&g, &x, &b, old, late_end);
    let e = Engine::new(Fault::Late, 1);
    assert!(evaluate(&late, &e, &mut || true).is_none());
    assert_eq!(e.calls.load(Ordering::SeqCst), 1);
}

fn cache() -> CachedLinearBounds {
    let mut c = CachedLinearBounds::default();
    let mut a = vec![-1.; N + 1];
    a[N] = -EPS;
    c.lower_a.insert(
        "relu2".into(),
        Array2::from_shape_vec((1, N + 1), a).unwrap(),
    );
    c
}

#[test]
fn spatial_feedback_default_owner_replays_original_row_once_and_keeps_unsafe_truth() {
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = end();
        let old = old(&g, &x, &b, end);
        let c = cache();
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let defaults = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            certified_root_cut_owner: Some(owner),
            certified_root_cut_raw_source: true,
            ..defaults
        };
        let e = Engine::new(Fault::None, 0);
        let mut bounds = [old];
        assert_eq!(
            try_tighten_with_spatial_advice(
                &config,
                &g,
                &x,
                &[vec![1.]],
                &[0.],
                &mut bounds,
                &b,
                None,
                Some(&e),
                Some(end),
                Some(Advice {
                    cache: &c,
                    active_indices: &[0]
                })
            ),
            Some(0)
        );
        assert!(bounds[0].0 > 0.249 + shift && bounds[0].0 <= 0.25 + shift);
        assert_eq!(bounds[0].1.to_bits(), old.1.to_bits());
        assert_eq!(e.calls.load(Ordering::SeqCst), 3);
        let mut again = [old];
        assert!(try_tighten_with_spatial_advice(
            &config,
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut again,
            &b,
            None,
            Some(&e),
            Some(end),
            Some(Advice {
                cache: &c,
                active_indices: &[0]
            })
        )
        .is_none());
        assert_eq!(again, [old]);
        assert_eq!(e.calls.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn spatial_feedback_cache_scores_cannot_publish_and_required_replay_refusal_is_atomic() {
    let (g, x, b) = fixture(-0.5);
    let end = end();
    let old = old(&g, &x, &b, end);
    for refuse in [false, true] {
        let mut c = cache();
        c.lower_a.get_mut("relu2").unwrap().fill(1e20);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let defaults = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            certified_root_cut_owner: Some(owner),
            certified_root_cut_raw_source: true,
            ..defaults
        };
        let e = Engine::new(if refuse { Fault::Refuse } else { Fault::None }, 3);
        let mut bounds = [old];
        let out = try_tighten_with_spatial_advice(
            &config,
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut bounds,
            &b,
            None,
            Some(&e),
            Some(end),
            Some(Advice {
                cache: &c,
                active_indices: &[0],
            }),
        );
        if refuse {
            assert!(out.is_none());
            assert_eq!(bounds, [old]);
        } else {
            assert_eq!(out, Some(0));
            assert!(bounds[0].0 <= -0.25);
        }
    }
}

#[test]
fn spatial_feedback_source_shape_selection_and_exact_prefix_quote_remain_required() {
    let (g, x, b) = fixture(0.);
    let end = end();
    let old = old(&g, &x, &b, end);
    for ranked in [
        vec![],
        vec![(0, 1.), (0, 1.)],
        vec![(N + 1, 1.)],
        vec![(0, f64::NAN)],
    ] {
        assert!(quote_rows(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "conv2",
            &ranked,
            8,
            MAX_WORK,
            MAX_SCRATCH,
            end,
            &mut || true
        )
        .is_none());
    }
    let mut q = quote(&g, &x, &b, old, end);
    q.prefix.output_dim += 1;
    let e = Engine::new(Fault::None, 0);
    assert!(evaluate(&q, &e, &mut || true).is_none());
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    let q = quote(&g, &x, &b, old, end);
    let e = Engine::new(Fault::None, 0);
    assert!(evaluate(&q, &e, &mut || false).is_none());
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn spatial_feedback_compact_objective_cache_permutations_and_upfront_scan_charge_match() {
    let (g, x, b) = fixture(0.);
    let end = end();
    let old = old(&g, &x, &b, end);
    let mut answer = None;
    for indices in [[0usize, 1], [1, 0]] {
        let mut cache = CachedLinearBounds::default();
        let mut a = Array2::zeros((2, N + 1));
        let cache_row = indices.iter().position(|i| *i == 1).unwrap();
        for i in 0..N {
            a[[cache_row, i]] = -1.;
        }
        cache.lower_a.insert("relu2".into(), a);
        let budget =
            WorkBudget::new(Some(end), MAX_SCRATCH, 0, MAX_WORK, 1, Duration::ZERO).unwrap();
        let mut lease = budget
            .lease(WorkRequest {
                scratch_bytes: QUOTE_SCRATCH,
                proposals: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        let engine = Engine::new(Fault::None, 0);
        let q = plan(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            1,
            2,
            Advice {
                cache: &cache,
                active_indices: &indices,
            },
            &engine,
            &mut lease,
            MAX_WORK,
            MAX_SCRATCH,
            end,
        )
        .unwrap();
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            PLAN_WORK + 256 * (N as u64 + 1)
        );
        assert_eq!(q.selected, N);
        let value = evaluate(&q, &engine, &mut || lease.checkpoint().is_ok())
            .unwrap()
            .lower;
        assert!(value > 0.249 && value <= 0.25);
        if let Some(previous) = answer {
            assert_eq!(value.to_bits(), previous);
        } else {
            answer = Some(value.to_bits());
        }
    }
    for indices in [[1usize, 1], [0, 2]] {
        let cache = cache();
        let budget =
            WorkBudget::new(Some(end), MAX_SCRATCH, 0, MAX_WORK, 1, Duration::ZERO).unwrap();
        let mut lease = budget
            .lease(WorkRequest {
                scratch_bytes: QUOTE_SCRATCH,
                ..WorkRequest::default()
            })
            .unwrap();
        let engine = Engine::new(Fault::None, 0);
        assert!(plan(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            1,
            2,
            Advice {
                cache: &cache,
                active_indices: &indices
            },
            &engine,
            &mut lease,
            MAX_WORK,
            MAX_SCRATCH,
            end
        )
        .is_none());
        assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn spatial_feedback_nondefault_original_alpha_is_preserved_across_private_map_replay() {
    let (g, x, b) = fixture(-0.5);
    let end = end();
    let old = old(&g, &x, &b, end);
    let mut alpha = GraphAlphaState::new();
    alpha.add_relu_node("relu1", &b["conv1"], false).unwrap();
    alpha.add_relu_node("relu2", &b["conv2"], false).unwrap();
    alpha.alphas.get_mut("relu2").unwrap().fill(0.25);
    let before = alpha.alpha("relu2").unwrap().clone();
    let ranked = (0..N).map(|i| (i, 1.)).collect::<Vec<_>>();
    let q = quote_rows(
        &g,
        &x,
        &b,
        Some(&alpha),
        &[1.],
        old,
        "conv2",
        &ranked,
        8,
        MAX_WORK,
        MAX_SCRATCH,
        end,
        &mut || true,
    )
    .unwrap();
    let lower = evaluate(&q, &Engine::new(Fault::None, 0), &mut || true)
        .unwrap()
        .lower;
    assert!(lower > -0.251 && lower <= -0.25);
    assert_eq!(alpha.alpha("relu2").unwrap(), &before);
    assert!(b["conv2"].lower().as_slice().unwrap()[0] < -1.99);
}

#[test]
fn spatial_feedback_advice_entry_preserves_nonspatial_original_cut_fallback() {
    use crate::network::raw_root_cut::raw_two_local_l1_root_cut_test_support::{
        baseline, fixture, BundleBackend,
    };
    struct Legacy {
        bundle: BundleBackend,
    }
    impl GemmEngine for Legacy {
        fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
            panic!("no ordinary GEMM")
        }
        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }
    impl GpuCrownBackward for Legacy {
        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }
        fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
            8
        }
        fn provides_bounded_two_local_l1_lower_cut(&self) -> bool {
            true
        }
        fn crown_backward_gpu(
            &self,
            _: &[GpuCrownLayer],
            _: &[f32],
            _: usize,
            _: &[f32],
            _: &[f32],
        ) -> Result<GpuCrownResult> {
            panic!("no ordinary CROWN")
        }
        fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
            &self,
            _: &[GpuResnetSegment],
            _: &GpuCrownSeed,
            _: &[f32],
            _: &[f32],
            _: &[Vec<f32>],
            _: &[Vec<f32>],
            _: Instant,
        ) -> Result<GpuCrownResult> {
            panic!("nonspatial source must keep its original cut route")
        }
        fn crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &self,
            s: &[GpuResnetSegment],
            q: &GpuCrownSeed,
            l: &[f32],
            u: &[f32],
            c: &ny_core::ResidentTwoLocalL1LowerCutCarrier,
            end: Instant,
        ) -> Result<f32> {
            self.bundle
                .crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(s, q, l, u, c, end)
        }
    }
    let (g, x, b) = fixture(0.);
    let end = end();
    let old = baseline(&g, &x, &b, end);
    let mut hint = CachedLinearBounds::default();
    hint.lower_a
        .insert("relu".into(), Array2::from_elem((1, 17), f32::NAN));
    let mut reference = None;
    for advised in [false, true] {
        let defaults = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            certified_root_cut_raw_source: true,
            certified_root_cut_owner: QueryOwner::new(Some(end), true),
            ..defaults
        };
        let engine = Legacy {
            bundle: BundleBackend::default(),
        };
        let mut out = [old];
        let got = try_tighten_with_spatial_advice(
            &config,
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut out,
            &b,
            None,
            Some(&engine),
            Some(end),
            advised.then_some(Advice {
                cache: &hint,
                active_indices: &[0],
            }),
        );
        assert_eq!(got, Some(0));
        assert_eq!(engine.bundle.bundle_calls.load(Ordering::SeqCst), 1);
        let bits = (out[0].0.to_bits(), out[0].1.to_bits());
        if let Some(previous) = reference {
            assert_eq!(bits, previous);
        } else {
            reference = Some(bits);
        }
    }
}
